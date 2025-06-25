import hashlib
import hmac
import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, Request, HTTPException, Header, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import os
import requests
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/whatsapp", tags=["Whatsapp Routes"])

# Configuration
WEBHOOK_VERIFY_TOKEN = os.getenv("VERIFY_TOKEN")
WEBHOOK_SECRET = os.getenv("APP_SECRET")
WHATSAPP_TOKEN = os.getenv("ACCESS_TOKEN")
WHATSAPP_PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID")

# Pydantic models for request validation
class WebhookMessage(BaseModel):
    id: str
    from_: str = Field(alias="from")
    timestamp: str
    type: str
    text: Optional[Dict[str, Any]] = None
    image: Optional[Dict[str, Any]] = None
    document: Optional[Dict[str, Any]] = None
    audio: Optional[Dict[str, Any]] = None
    video: Optional[Dict[str, Any]] = None

class WebhookContact(BaseModel):
    profile: Dict[str, str]
    wa_id: str

class WebhookMetadata(BaseModel):
    display_phone_number: str
    phone_number_id: str

class WebhookValue(BaseModel):
    messaging_product: str
    metadata: WebhookMetadata
    contacts: Optional[list[WebhookContact]] = None
    messages: Optional[list[WebhookMessage]] = None
    statuses: Optional[list[Dict[str, Any]]] = None

class WebhookEntry(BaseModel):
    id: str
    changes: list[Dict[str, Any]]

class WebhookPayload(BaseModel):
    object: str
    entry: list[WebhookEntry]

def verify_webhook_signature(payload: bytes, signature: str) -> bool:
    """Verify the webhook signature from WhatsApp"""
    if not signature.startswith("sha256="):
        return False
    
    expected_signature = hmac.new(
        WEBHOOK_SECRET.encode(),
        payload,
        hashlib.sha256
    ).hexdigest()
    
    received_signature = signature[7:]  # Remove 'sha256=' prefix
    return hmac.compare_digest(expected_signature, received_signature)

async def send_whatsapp_message(to_phone_number: str, message_text: str, phone_number_id: str = None):
    """Send a WhatsApp message using the WhatsApp Business API"""
    if not WHATSAPP_TOKEN or WHATSAPP_TOKEN == "your-whatsapp-token-here":
        logger.warning("WhatsApp token not configured, cannot send messages")
        return False
    
    phone_id = phone_number_id or WHATSAPP_PHONE_NUMBER_ID
    if not phone_id or phone_id == "your-phone-number-id":
        logger.warning("WhatsApp phone number ID not configured, cannot send messages")
        return False
    
    url = f"https://graph.facebook.com/v18.0/{phone_id}/messages"
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "messaging_product": "whatsapp",
        "to": to_phone_number,
        "type": "text",
        "text": {
            "body": message_text
        }
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        logger.info(f"Message sent successfully to {to_phone_number}")
        return True
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to send message to {to_phone_number}: {e}")
        return False

async def process_webhook_message(message: WebhookMessage, contact: WebhookContact, metadata: WebhookMetadata):
    """Process incoming WhatsApp message and respond with uppercase version"""
    logger.info(f"Processing message from {contact.wa_id}: {message.id}")
    
    # Handle text messages - convert to uppercase and send back
    if message.type == "text" and message.text:
        text_content = message.text.get("body", "")
        logger.info(f"Text message received: {text_content}")
        
        # Convert to uppercase
        uppercase_text = text_content.upper()
        
        # Send the uppercase message back
        await send_whatsapp_message(
            to_phone_number=contact.wa_id,
            message_text=uppercase_text,
            phone_number_id=metadata.phone_number_id
        )
        
    elif message.type == "image" and message.image:
        logger.info(f"Image message received: {message.image.get('id', 'Unknown')}")
        # For non-text messages, send a response indicating we received it
        await send_whatsapp_message(
            to_phone_number=contact.wa_id,
            message_text="IMAGE RECEIVED - I CAN ONLY CONVERT TEXT MESSAGES TO UPPERCASE!",
            phone_number_id=metadata.phone_number_id
        )
        
    elif message.type == "document" and message.document:
        logger.info(f"Document message received: {message.document.get('id', 'Unknown')}")
        await send_whatsapp_message(
            to_phone_number=contact.wa_id,
            message_text="DOCUMENT RECEIVED - I CAN ONLY CONVERT TEXT MESSAGES TO UPPERCASE!",
            phone_number_id=metadata.phone_number_id
        )
        
    elif message.type == "audio" and message.audio:
        logger.info(f"Audio message received: {message.audio.get('id', 'Unknown')}")
        await send_whatsapp_message(
            to_phone_number=contact.wa_id,
            message_text="AUDIO RECEIVED - I CAN ONLY CONVERT TEXT MESSAGES TO UPPERCASE!",
            phone_number_id=metadata.phone_number_id
        )
        
    elif message.type == "video" and message.video:
        logger.info(f"Video message received: {message.video.get('id', 'Unknown')}")
        await send_whatsapp_message(
            to_phone_number=contact.wa_id,
            message_text="VIDEO RECEIVED - I CAN ONLY CONVERT TEXT MESSAGES TO UPPERCASE!",
            phone_number_id=metadata.phone_number_id
        )
    
    else:
        # Handle other message types
        await send_whatsapp_message(
            to_phone_number=contact.wa_id,
            message_text="MESSAGE RECEIVED - I CAN ONLY CONVERT TEXT MESSAGES TO UPPERCASE!",
            phone_number_id=metadata.phone_number_id
        )
    
async def process_webhook_status(status: Dict[str, Any]):
    """Process message status updates"""
    logger.info(f"Status update: {status}")
    
    # Handle delivery receipts, read receipts, etc.
    message_id = status.get("id")
    status_type = status.get("status")
    
    if status_type == "delivered":
        logger.info(f"Message {message_id} was delivered")
    elif status_type == "read":
        logger.info(f"Message {message_id} was read")
    elif status_type == "failed":
        logger.error(f"Message {message_id} failed to deliver")

async def handle_webhook_data(webhook_data: WebhookPayload, background_tasks: BackgroundTasks):
    """Handle the webhook payload"""
    for entry in webhook_data.entry:
        for change in entry.changes:
            if change.get("field") == "messages":
                value_data = change.get("value", {})
                
                # Get metadata
                metadata_data = value_data.get("metadata", {})
                metadata = WebhookMetadata(**metadata_data)
                
                # Process messages
                messages = value_data.get("messages", [])
                contacts = value_data.get("contacts", [])
                
                # Create contact lookup for efficiency
                contact_lookup = {contact["wa_id"]: contact for contact in contacts}
                
                for message_data in messages:
                    try:
                        message = WebhookMessage(**message_data)
                        contact_info = contact_lookup.get(message.from_)
                        
                        if contact_info:
                            contact = WebhookContact(**contact_info)
                            background_tasks.add_task(process_webhook_message, message, contact, metadata)
                    except Exception as e:
                        logger.error(f"Error processing message: {e}")
                
                # Process status updates
                statuses = value_data.get("statuses", [])
                for status in statuses:
                    background_tasks.add_task(process_webhook_status, status)

@router.get("/webhook")
async def verify_webhook(
    hub_mode: str = Query(None, alias="hub.mode"),
    hub_challenge: str = Query(None, alias="hub.challenge"),
    hub_verify_token: str = Query(None, alias="hub.verify_token")
):
    """
    Webhook verification endpoint for WhatsApp Business API
    This endpoint is called by WhatsApp to verify the webhook URL
    """
    
    if (hub_mode == "subscribe" and 
        hub_verify_token == WEBHOOK_VERIFY_TOKEN):
        logger.info("Webhook verified successfully")
        return int(hub_challenge)
    else:
        logger.warning("Webhook verification failed")
        raise HTTPException(status_code=403, detail="Verification failed")

@router.post("/webhook")
async def receive_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    x_hub_signature_256: Optional[str] = Header(None, alias="X-Hub-Signature-256")
):
    """
    Main webhook endpoint to receive WhatsApp messages and status updates
    """
    try:
        # Get raw request body for signature verification
        body = await request.body()
        
        # Verify webhook signature if secret is configured
        if WEBHOOK_SECRET:
            if not x_hub_signature_256:
                raise HTTPException(status_code=401, detail="Missing signature")
            
            if not verify_webhook_signature(body, x_hub_signature_256):
                raise HTTPException(status_code=401, detail="Invalid signature")
        
        # Parse JSON payload
        try:
            payload_dict = await request.json()
            webhook_payload = WebhookPayload(**payload_dict)
        except Exception as e:
            logger.error(f"Error parsing webhook payload: {e}")
            raise HTTPException(status_code=400, detail="Invalid JSON payload")
        
        # Process webhook data in background
        await handle_webhook_data(webhook_payload, background_tasks)
        
        # Return success response
        return JSONResponse(
            status_code=200,
            content={"status": "success", "message": "Webhook received"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing webhook: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "WhatsApp Webhook"
    }

@router.get("/")
async def root():
    """Root endpoint with basic information"""
    return {
        "message": "WhatsApp Webhook API - Uppercase Bot",
        "version": "1.0.0",
        "description": "Converts all text messages to uppercase and sends them back",
        "endpoints": {
            "webhook_verify": "GET /webhook",
            "webhook_receive": "POST /webhook",
            "health": "GET /health"
        }
    }