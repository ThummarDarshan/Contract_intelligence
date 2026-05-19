from fastapi import APIRouter
from api.upload import router as upload_router
from api.status import router as status_router
from api.analyze import router as analyze_router

router = APIRouter()
router.include_router(upload_router)
router.include_router(status_router)
router.include_router(analyze_router)