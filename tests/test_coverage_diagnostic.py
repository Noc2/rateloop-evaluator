from copy import deepcopy
from types import SimpleNamespace
import pytest
from rateloop_evaluator.backends import tokenizer_commitment
from rateloop_evaluator.coverage_diagnostic import token_coverage_diagnostic
from rateloop_evaluator.general_benchmark import freeze_benchmark
from rateloop_evaluator.general_qualification import freeze_operating_point, test_representatives
from test_general_benchmark import row, SOURCES


class Backend:
    manifest = {'files': {'model.safetensors': 'a'*64, 'tokenizer.json': 'b'*64}}
    adapter_version = 1
    def load(self):
        return SimpleNamespace(config=SimpleNamespace(max_len=4096), encoder=SimpleNamespace(config=SimpleNamespace(position_biased_input=False)))
    def count_tokens(self, text, questions): return 513
    def predict(self, *_): pytest.fail('Coverage diagnostic must not perform inference')
    def unload(self): self.unloaded = True


def test_coverage_keeps_all_final_groups_and_marks_theoretical_model_overflow():
    rows = [row(i) for i in range(60)]
    for value in rows: value['template']['maxTokens'] = 512
    manifest = freeze_benchmark(rows, SOURCES)
    backend = Backend()
    point = freeze_operating_point(manifest, model={'weights_sha256':'a'*64,
        'tokenizer_sha256':tokenizer_commitment(backend.manifest).removeprefix('sha256:'),
        'adapter':'gliner','runtime':'test','quantization':'float32'},
        route={'id':'test','version':1,'evidence_policy':'supplied-only'})
    before = deepcopy(rows)
    report = token_coverage_diagnostic(manifest, rows, point, backend)
    assert rows == before and backend.unloaded
    count = len(test_representatives(manifest, rows))
    assert report['total_groups'] == count and report['current_template_supported_groups'] == 0
    assert [item['within_budget_groups'] for item in report['budgets']] == [0,count,count,count,count]
    assert report['budgets'][-1]['within_architectural_limit'] is False
    assert report['inference_performed'] is report['quality_gate'] is report['activation_changed'] is False
    wrong = Backend(); wrong.adapter_version = 2
    with pytest.raises(ValueError, match='original frozen'): token_coverage_diagnostic(manifest,rows,point,wrong)
    wrong = Backend(); wrong.manifest = {'files':{**Backend.manifest['files'],'model.safetensors':'c'*64}}
    with pytest.raises(ValueError, match='original frozen'): token_coverage_diagnostic(manifest,rows,point,wrong)
