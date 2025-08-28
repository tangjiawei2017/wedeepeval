from typing import Any, Dict, Optional


def success(data: Optional[Any] = None, message: str = "ok", code: int = 0) -> Dict[str, Any]:
    return {
        "code": code,
        "message": message,
        "data": data,
    }


def error(message: str, code: int = 1, data: Optional[Any] = None) -> Dict[str, Any]:
    return {
        "code": code,
        "message": message,
        "data": data,
    }

