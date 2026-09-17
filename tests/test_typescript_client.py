import json
import subprocess
from pathlib import Path
from rateloop_evaluator.protocol import EvaluationRequest, make_result

ROOT = Path(__file__).resolve().parents[1]

def test_client_binds_local_response_and_rejects_remote_plaintext():
    request = EvaluationRequest.model_validate_json((ROOT / "examples/reply-request.json").read_text())
    result = make_result(workspaceId=request.workspaceId, caseId=request.caseId, modelBundleId=request.modelBundleId,
        inputCommitment=request.input_commitment(), templateCommitment=request.template_commitment(),
        outcome="uncertain", abstainReason="uncalibrated", criteria=[], durationMs=1, observedAt="2026-09-17T12:00:00.000Z")
    source = """
import assert from 'node:assert/strict';
import {EvaluatorClient} from './clients/typescript/index.ts';
const [request,result] = JSON.parse(process.argv[1]);
assert.throws(()=>new EvaluatorClient({endpoint:'http://remote.example',token:'test'}));
let calls = 0;
const client = new EvaluatorClient({endpoint:'http://127.0.0.1:8765',token:'test',fetch:async(url,options)=>{
  calls++; assert.equal(options.redirect,'error'); assert.equal(options.headers.Authorization,'Bearer test');
  return new Response(JSON.stringify(result));
}});
assert.deepEqual(await client.evaluate(request),result);
await assert.rejects(client.evaluate({...request,caseId:'different'}),/different case/);
assert.equal(calls,2);
"""
    subprocess.run(["node","--experimental-strip-types","--input-type=module","-e",source,json.dumps([request.model_dump(),result.model_dump()])],cwd=ROOT,check=True,capture_output=True,text=True)
