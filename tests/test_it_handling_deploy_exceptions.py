from pathlib import Path


def test_gracefully_handle_deploy_exception(client, deploy_pipeline):
    pipeline_name = "broken_rag_pipeline"
    pipeline_def = (Path(__file__).parent / "test_files/yaml" / "broken_rag_pipeline.yml").read_text()

    deploy_response = deploy_pipeline(client, pipeline_name, pipeline_def)
    # NOTE: The deprecated deploy method returns 409 for any ValueError (incorrectly)
    #       We can simply check that the status code is not 200.
    assert deploy_response.status_code != 200
    assert "Couldn't deserialize component 'llm'" in deploy_response.json()["detail"]
