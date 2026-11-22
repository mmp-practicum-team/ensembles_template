"""Validate route locations and annotations required by the assignment."""

import inspect

from fastapi.routing import APIRoute

from ensembles.backend.app import app


errors = []
for route in app.routes:
    if not isinstance(route, APIRoute):
        continue
    if getattr(route.endpoint, "__module__", None) != "ensembles.backend.app":
        errors.append(f"{route.path}: обработчик должен быть определён в ensembles/backend/app.py")
    signature = inspect.signature(route.endpoint)
    for name, param in signature.parameters.items():
        if param.annotation is inspect.Signature.empty:
            errors.append(f"{route.path}: аргумент {name} без аннотации")
    if signature.return_annotation is inspect.Signature.empty and route.response_model is None:
        errors.append(f"{route.path}: отсутствует аннотация результата или response_model")

if errors:
    raise SystemExit("\n".join(errors))
print("OK: обработчики находятся в app.py, аннотации аргументов и результатов присутствуют.")
