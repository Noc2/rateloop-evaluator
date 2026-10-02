"""Both fenced inference queues require the deployed terminal acknowledgement."""
import json

import httpx
import pytest

from rateloop_evaluator.connector import ConnectorUnavailable
from rateloop_evaluator.native_chat_pool import NativeChatPool
from test_connector import setup
from test_native_chat_pool import Backend, fixture
from test_worker import website


@pytest.mark.parametrize("consumer",["native","retained"])
@pytest.mark.parametrize("invalid",[{}, {"completed":False}, {"completed":"true"}, {"completed":1}])
def test_both_queues_preserve_original_receipt_and_lease_until_explicit_completion(consumer,invalid,website):
    if consumer=="retained":
        worker,_request,backend,_remote,behavior,calls=website
        behavior["complete_body"]=invalid
        with pytest.raises(ConnectorUnavailable): worker.run_once()
        assert worker._saved() is not None and backend.calls==1
        behavior["complete_body"]={"completed":True,"replayed":True}
        assert worker.run_once()["state"]=="completed"
        assert worker._saved() is None and backend.calls==1
        completions=[json.loads(request.content) for request in calls if request.url.path.endswith("/complete")]
    else:
        job=fixture(); backend=Backend(); completions=[]
        def transport(request):
            body=json.loads(request.content); action=body["action"]
            if action=="claim": return httpx.Response(200,json={"job":job})
            if action=="heartbeat_job": return httpx.Response(200,json={"leaseExpiresAt":job["leaseExpiresAt"]})
            if action=="complete":
                completions.append(body)
                return httpx.Response(200,json=invalid if len(completions)==1 else {"completed":True,"replayed":True})
            raise AssertionError(action)
        worker=NativeChatPool(secret="s"*32,base_url="https://www.rateloop.ai",bundles=[job["baseRegistration"]],backend=backend,
            transport=httpx.MockTransport(transport))
        try:
            with pytest.raises(ConnectorUnavailable): worker.run_once()
            assert worker.pending is not None and backend.calls==1
            assert worker.run_once()=={"state":"completed"}
            assert worker.pending is None and backend.calls==1
        finally: worker.close()
    assert len(completions)==2 and completions[0]==completions[1]
