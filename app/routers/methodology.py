from pathlib import Path
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

router = APIRouter()
templates = Jinja2Templates(directory=Path("app/templates"))


@router.get("/methodology", response_class=HTMLResponse)
async def methodology_page(request: Request):
    return templates.TemplateResponse("methodology.html", {"request": request})
