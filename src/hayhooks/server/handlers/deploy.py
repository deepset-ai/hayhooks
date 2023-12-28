from fastapi import HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, create_model

from hayhooks.server import app
from hayhooks.server.pipelines import registry


class PipelineDefinition(BaseModel):
    name: str
    source_code: str


@app.post("/deploy")
async def deploy(pipeline_def: PipelineDefinition):
    try:
        pipe = registry.add(pipeline_def.name, pipeline_def.source_code)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=f"{e}") from e

    request_model = {}
    for component_name, inputs in pipe.inputs().items():
        # Inputs have this form:
        # {
        #     'first_addition': { <-- Component Name
        #         'value': {'type': <class 'int'>, 'is_mandatory': True}, <-- Input
        #         'add': {'type': typing.Optional[int], 'is_mandatory': False, 'default_value': None}, <-- Input
        #     },
        #     'second_addition': {'add': {'type': typing.Optional[int], 'is_mandatory': False}},
        # }
        component_model = {}
        for name, typedef in inputs.items():
            component_model[name] = (typedef["type"], typedef.get("default_value", ...))
        request_model[component_name] = (create_model('ComponentParams', **component_model), ...)

    PipelineRunRequest = create_model(f'{pipeline_def.name.capitalize()}RunRequest', **request_model)

    response_model = {}
    for component_name, outputs in pipe.outputs().items():
        # Outputs have this form:
        # {
        #   'second_addition': { <-- Component Name
        #       'result': {'type': "<class 'int'>"}  <-- Output
        #   },
        # }
        component_model = {}
        for name, typedef in outputs.items():
            component_model[name] = (typedef["type"], ...)
        response_model[component_name] = (create_model('ComponentParams', **component_model), ...)

    PipelineRunResponse = create_model(f'{pipeline_def.name.capitalize()}RunResponse', **response_model)

    # There's no way in FastAPI to define the type of the request body other than annotating
    # the endpoint handler. We have to ignore the type here to make FastAPI happy while
    # silencing static type checkers (that would have good reasons to trigger!).
    async def pipeline_run(pipeline_run_req: PipelineRunRequest) -> JSONResponse:  # type: ignore
        output = pipe.run(data=pipeline_run_req.dict())
        return JSONResponse(PipelineRunResponse(**output).model_dump(), status_code=200)

    app.add_api_route(
        path=f"/{pipeline_def.name}",
        endpoint=pipeline_run,
        methods=["POST"],
        name=pipeline_def.name,
        response_model=PipelineRunResponse,
    )
    app.openapi_schema = None
    app.setup()

    return {"name": pipeline_def.name}
