import json

import pytest

from hello_world import app


@pytest.fixture()
def api_event():
    """ Generates API Event"""

    return {
      "text": "TSLA is going to the moon. I think TSLA is the greatest company ever and GM and other car manufacturers don't stand a chance when competing with TSLA"
    }


def test_lambda_handler(api_event, mocker):

    ret = app.lambda_handler(api_event, "")
    data = json.loads(ret["body"])

    assert ret["statusCode"] == 200
    assert len(data) == 2
    assert 'ticker' in data[0].dict_keys()
    assert 'sentiment_score' in data[0].dict_keys()
