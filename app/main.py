from fastapi import FastAPI
from .routes import whatsapp_routes

app = FastAPI(
    title="WhatsApp Webhook API",
    description="Professional WhatsApp Business API webhook endpoint",
    version="1.0.0"
)

# Include routers
app.include_router(whatsapp_routes.router)