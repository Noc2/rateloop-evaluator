"""Token-budget diagnostics on an unchanged frozen final cohort, without inference."""
from .backends import render_input, model_token_limit, tokenizer_commitment
from .general_benchmark import verify_manifest, digest
from .general_qualification import test_representatives, verify_operating_point


def token_coverage_diagnostic(manifest, rows, point, backend, budgets=(512, 1024, 2048, 4096, 8192)):
    verify_manifest(manifest, rows)
    verify_operating_point(manifest, point)
    if not budgets or any(type(value) is not int or value < 1 for value in budgets):
        raise ValueError('Coverage budgets must be positive integer token counts')
    model = backend.load()
    identity = backend.manifest
    if (identity['files']['model.safetensors'] != point['model']['weights_sha256']
            or tokenizer_commitment(identity).removeprefix('sha256:') != point['model']['tokenizer_sha256']
            or point['model']['adapter'] != 'gliner' or backend.adapter_version != 1):
        raise ValueError('Coverage diagnostic requires the original frozen model, tokenizer and adapter')
    architectural_limit = model_token_limit(model)
    measured = []
    try:
        for entry, row in test_representatives(manifest, rows):
            count = backend.count_tokens(render_input(row['input']), row['template']['questions'])
            measured.append((count, row['template']['maxTokens'], row['labels']))
        def summarize(cap):
            supported = [item for item in measured if item[0] <= cap]
            return {'budget_tokens': cap, 'within_budget_groups': len(supported),
                'total_groups': len(measured), 'token_coverage': len(supported)/len(measured),
                'within_architectural_limit': cap <= architectural_limit,
                'labels': {qid: {label: {'support': sum(labels[qid] == label for _,_,labels in measured),
                    'within_budget': sum(labels[qid] == label for _,_,labels in supported)}
                    for label in sorted({labels[qid] for _,_,labels in measured})}
                    for qid in measured[0][2]}}
        return {'schema_version': 'rateloop.token-coverage-diagnostic.v1',
            'benchmark_commitment': manifest['commitment'], 'original_operating_point_commitment': point['commitment'],
            'model': point['model'], 'architectural_limit_tokens': architectural_limit,
            'token_count_commitment': digest([count for count,_,_ in measured]),
            'total_groups': len(measured), 'current_template_supported_groups': sum(count <= cap for count,cap,_ in measured),
            'budgets': [summarize(cap) for cap in budgets], 'quality_gate': False, 'inference_performed': False,
            'activation_changed': False,
            'limits': 'Re-tokenization of the original frozen final cohort with original weights, tokenizer, rubric and v1 adapter. '
                'Budget coverage is not accuracy or calibrated decision coverage. Budgets above the architectural limit are descriptive only. '
                'The deployed 512-token template is unchanged; no truncation, model execution, new labels, training or activation.'}
    finally:
        backend.unload()
