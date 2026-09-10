from fastapi import APIRouter, HTTPException, Request

router = APIRouter()


@router.post("/telegram/webhook")
async def telegram_webhook(request: Request):
    settings = request.app.state.settings
    if request.headers.get("X-Telegram-Bot-Api-Secret-Token") != settings.telegram_webhook_secret:
        raise HTTPException(status_code=403, detail="Bad webhook secret")
    ptb_app = getattr(request.app.state, "ptb_app", None)
    if ptb_app is None:
        raise HTTPException(status_code=503, detail="Bot disabled")
    from telegram import Update

    update = Update.de_json(await request.json(), ptb_app.bot)
    await ptb_app.process_update(update)
    return {}
