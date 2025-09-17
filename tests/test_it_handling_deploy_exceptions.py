from pathlib import Path


def test_gracefully_handle_deploy_exception(client, deploy_yaml_pipeline):
    pipeline_name = "broken_rag_pipeline"
    pipeline_source_code = (Path(__file__).parent / "test_files/yaml/broken/broken_rag_pipeline.yml").read_text()

    deploy_response = deploy_yaml_pipeline(client, pipeline_name, pipeline_source_code)
    assert deploy_response.status_code == 500
    assert "Couldn't deserialize component 'llm'" in deploy_response.json()["detail"]
