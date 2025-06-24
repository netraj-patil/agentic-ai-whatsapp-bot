import logging
from datetime import datetime
from functools import wraps
from typing import List, Dict, Any, Union, Optional

from langchain.tools import tool
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from langchain_google_community.calendar.create_event import CalendarCreateEvent
from langchain_google_community.calendar.search_events import CalendarSearchEvents
from langchain_google_community.calendar.update_event import CalendarUpdateEvent
from langchain_google_community.calendar.delete_event import CalendarDeleteEvent
import pytz
from pydantic import BaseModel, Field, validator
import os

# ==================== CONFIGURATION ====================

# Calendar configuration
CALENDAR_ID = os.getenv("CALENDAR_ID")
SCOPES = ["https://www.googleapis.com/auth/calendar"]

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('calendar_tools.log'),
        logging.StreamHandler()  # Also log to console
    ]
)

# Create a logger specific to this module
logger = logging.getLogger(__name__)

# ==================== GOOGLE CALENDAR SETUP ====================

try:
    # Load service account credentials from JSON file
    creds = Credentials.from_service_account_file(
        "service-account.json",
        scopes=SCOPES,
    )
    logger.info("Successfully loaded service account credentials")
except Exception as e:
    logger.error(f"Failed to load service account credentials: {str(e)}")
    raise

try:
    # Create the Google Calendar API service instance
    calendar_service: "googleapiclient.discovery.Resource" = build(
        serviceName="calendar",
        version="v3",
        credentials=creds,
        cache_discovery=False,  # Disable local discovery cache for server environments
    )
    logger.info("Successfully initialized Google Calendar API service")
except Exception as e:
    logger.error(f"Failed to initialize Google Calendar API service: {str(e)}")
    raise

# ==================== UTILITY FUNCTIONS ====================

def log_function_call(func):
    """
    Decorator to log function calls, parameters, and results.
    
    Args:
        func: The function to be decorated
        
    Returns:
        Wrapped function with logging capabilities
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        
        # Log function call with sanitized parameters (avoid logging sensitive data)
        sanitized_kwargs = {k: v for k, v in kwargs.items() if k not in ['credentials', 'api_key']}
        logger.info(f"Calling {func_name} with args: {args}, kwargs: {sanitized_kwargs}")
        
        try:
            # Execute the function
            result = func(*args, **kwargs)
            logger.info(f"{func_name} completed successfully")
            return result
        except Exception as e:
            logger.error(f"{func_name} failed with error: {str(e)}", exc_info=True)
            raise
    
    return wrapper

def parse_datetime_string(dt_str: str) -> datetime:
    """
    Parse datetime strings in various common formats.
    
    This function attempts to parse datetime strings using multiple formats
    commonly used in calendar applications and APIs.
    
    Args:
        dt_str: A datetime string in various possible formats
        
    Returns:
        A datetime object parsed from the input string
        
    Raises:
        ValueError: If the datetime string cannot be parsed in any supported format
        
    Examples:
        >>> parse_datetime_string("2023-12-25T15:30:00Z")
        datetime(2023, 12, 25, 15, 30, tzinfo=timezone.utc)
        
        >>> parse_datetime_string("2023-12-25 15:30:00")
        datetime(2023, 12, 25, 15, 30)
    """
    logger.debug(f"Attempting to parse datetime string: {dt_str}")
    
    try:
        # Try ISO format first (standard format used by most APIs)
        parsed_dt = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
        logger.debug(f"Successfully parsed using ISO format: {parsed_dt}")
        return parsed_dt
    except ValueError:
        logger.debug("ISO format parsing failed, trying alternative formats")
        
        # List of alternative datetime formats to try
        formats = [
            "%Y-%m-%d %H:%M:%S",    # 2023-12-25 15:30:00
            "%Y-%m-%dT%H:%M:%S",    # 2023-12-25T15:30:00
            "%Y-%m-%d %H:%M",       # 2023-12-25 15:30
            "%Y-%m-%dT%H:%M",       # 2023-12-25T15:30
        ]
        
        for fmt in formats:
            try:
                parsed_dt = datetime.strptime(dt_str, fmt)
                logger.debug(f"Successfully parsed using format '{fmt}': {parsed_dt}")
                return parsed_dt
            except ValueError:
                continue
        
        # If all parsing attempts fail, raise an error
        error_msg = f"Unable to parse datetime: {dt_str}. Supported formats: ISO, YYYY-MM-DD HH:MM:SS, YYYY-MM-DDTHH:MM:SS, etc."
        logger.error(error_msg)
        raise ValueError(error_msg)

# ==================== PYDANTIC MODELS ====================

class CreateCalEventInput(BaseModel):
    """
    Input model for creating calendar events.
    
    Attributes:
        summary: The event title/summary
        start_datetime: Event start time as ISO string
        end_datetime: Event end time as ISO string
        description: Optional event description
        reminders: Reminder configuration - None, boolean, or list of reminder dicts
    """
    summary: str = Field(..., description="The title of the event")
    start_datetime: str = Field(..., description="Event start time (ISO format)")
    end_datetime: str = Field(..., description="Event end time (ISO format)")
    description: str = Field(default="", description="Optional event description")
    reminders: Union[None, bool, List[Dict[str, Any]]] = Field(
        default=None, 
        description="Reminder settings - None, boolean for defaults, or list of reminder configs"
    )

class SearchCalEventsInput(BaseModel):
    """
    Input model for searching calendar events.
    
    Attributes:
        min_datetime: Start of search time window
        max_datetime: End of search time window
        max_results: Maximum number of events to return
        order_by: Sort order for results
        query: Optional search query string
        single_events: Whether to expand recurring events
    """
    min_datetime: str = Field(..., description="Start of search window (ISO format)")
    max_datetime: str = Field(..., description="End of search window (ISO format)")
    max_results: Optional[int] = Field(default=10, description="Maximum events to return")
    order_by: Optional[str] = Field(default="startTime", description="Sort order: 'startTime' or 'updated'")
    query: Optional[str] = Field(default=None, description="Search query string")
    single_events: Optional[bool] = Field(default=True, description="Expand recurring events")

class UpdateCalEventInput(BaseModel):
    """
    Input model for updating calendar events.
    
    Attributes:
        event_id: ID of the event to update
        summary: New event title
        description: New event description
        start_datetime: New start time
        end_datetime: New end time
        location: New event location
        attendees: List of attendee email addresses
        reminders: New reminder configuration
        send_updates: Whether to notify attendees of changes
        timezone: Event timezone
    """
    event_id: str = Field(..., description="ID of the event to update")
    summary: Optional[str] = Field(default=None, description="New event title")
    description: Optional[str] = Field(default=None, description="New event description")
    start_datetime: Optional[datetime] = Field(default=None, description="New start time")
    end_datetime: Optional[datetime] = Field(default=None, description="New end time")
    location: Optional[str] = Field(default=None, description="New event location")
    attendees: Optional[List[str]] = Field(default=None, description="List of attendee emails")
    reminders: Union[None, bool, List[Dict[str, Any]]] = Field(default=None, description="Reminder settings")
    send_updates: Optional[str] = Field(default=None, description="Notify attendees: 'all', 'externalOnly', 'none'")
    timezone: Optional[str] = Field(default="Asia/Kolkata", description="Event timezone")

class DeleteCalEventInput(BaseModel):
    """
    Input model for deleting calendar events.
    
    Attributes:
        event_id: ID of the event to delete
        send_updates: Whether to notify attendees of deletion
    """
    event_id: str = Field(..., description="ID of the event to delete")
    send_updates: Optional[str] = Field(default=None, description="Notify attendees: 'all', 'externalOnly', 'none'")

# ==================== MAIN TOOL FUNCTIONS ====================

def get_all_calendar_tools():
    """
    Returns a comprehensive list of all Google Calendar management tools.
    
    This function provides access to all available calendar operations including
    creating, searching, updating, and deleting events, as well as utility functions.
    
    Returns:
        List of LangChain tools for Google Calendar operations
        
    Tools included:
        - createCalEvent: Create new calendar events
        - searchCalEvents: Search for existing events
        - updateCalEvent: Update existing events
        - deleteCalEvent: Delete events
        - getCurrentTimeInfo: Get current time and timezone information
    """
    logger.info("Retrieving all calendar tools")
    tools = [
        createCalEvent,
        searchCalEvents,
        updateCalEvent,
        deleteCalEvent,
        getCurrentTimeInfo,
    ]
    logger.info(f"Returning {len(tools)} calendar tools")
    return tools

@tool(args_schema=CreateCalEventInput)
@log_function_call
def createCalEvent(
    summary: str,
    start_datetime: str,
    end_datetime: str,
    description: str = "",
    reminders: Union[None, bool, List[Dict[str, Any]]] = None
) -> str:
    """
    Create a new event in Google Calendar.
    
    This function creates a new calendar event with the specified details.
    It handles datetime parsing and timezone conversion automatically.
    
    Args:
        summary: The title/summary of the event
        start_datetime: Event start time as a string (various formats supported)
        end_datetime: Event end time as a string (various formats supported)
        description: Optional detailed description of the event
        reminders: Reminder configuration options:
                  - None: No reminders
                  - True: Use default reminders
                  - List of dicts: Custom reminders like [{'method': 'email', 'minutes': 10}]
    
    Returns:
        JSON string containing the API response with event details
        
    Raises:
        ValueError: If datetime strings cannot be parsed
        
    Example:
        >>> createCalEvent(
        ...     summary="Team Meeting",
        ...     start_datetime="2023-12-25T10:00:00",
        ...     end_datetime="2023-12-25T11:00:00",
        ...     description="Weekly team sync",
        ...     reminders=[{'method': 'email', 'minutes': 15}]
        ... )
    """
    logger.info(f"Creating calendar event: {summary}")
    
    try:
        # Parse datetime strings into datetime objects
        start_dt = parse_datetime_string(start_datetime)
        end_dt = parse_datetime_string(end_datetime)
        logger.debug(f"Parsed start time: {start_dt}, end time: {end_dt}")
        
        # Validate that end time is after start time
        if end_dt <= start_dt:
            error_msg = f"End time ({end_dt}) must be after start time ({start_dt})"
            logger.error(error_msg)
            return f"Error: {error_msg}"
        
    except ValueError as e:
        logger.error(f"DateTime parsing error: {str(e)}")
        return f"Error parsing datetime: {str(e)}"
    
    try:
        # Initialize the calendar create tool
        tool = CalendarCreateEvent(api_resource=calendar_service)
        
        # Prepare the event payload
        event_payload = {
            "summary": summary,
            "start_datetime": start_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "end_datetime": end_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "timezone": "Asia/Kolkata",
            "description": description,
            "reminders": reminders,
            "calendar_id": CALENDAR_ID
        }
        
        logger.debug(f"Event payload: {event_payload}")
        
        # Create the event
        response = tool.invoke(event_payload)
        logger.info(f"Successfully created event: {summary}")
        
        return str(response)
        
    except Exception as e:
        logger.error(f"Failed to create calendar event: {str(e)}", exc_info=True)
        return f"Error creating event: {str(e)}"

@tool(args_schema=SearchCalEventsInput)
@log_function_call
def searchCalEvents(
    min_datetime: str,
    max_datetime: str,
    max_results: int = 10,
    order_by: str = "startTime",
    query: Optional[str] = None,
    single_events: bool = True,
) -> str:
    """
    Search for events in Google Calendar within a specified time range.
    
    This function searches for calendar events based on various criteria including
    time range, search terms, and result ordering options.
    
    Args:
        min_datetime: Start of the search time window (ISO format string)
        max_datetime: End of the search time window (ISO format string)
        max_results: Maximum number of events to return (default: 10)
        order_by: Sort order for results - 'startTime' or 'updated' (default: 'startTime')
        query: Optional search string to filter events by title/description
        single_events: If True, expand recurring events into individual instances
    
    Returns:
        JSON string containing matching events with their details
        
    Example:
        >>> searchCalEvents(
        ...     min_datetime="2023-12-20T00:00:00",
        ...     max_datetime="2023-12-31T23:59:59",
        ...     max_results=5,
        ...     query="meeting"
        ... )
    """
    logger.info(f"Searching calendar events from {min_datetime} to {max_datetime}")
    if query:
        logger.info(f"Search query: {query}")
    
    # Calendar information for the search tool
    calendars_info = f"""[
      {{
        "id": "{CALENDAR_ID}",
        "summary": "Primary Calendar"
      }}
    ]"""

    try:
        # Parse datetime strings
        min_dt = parse_datetime_string(min_datetime)
        max_dt = parse_datetime_string(max_datetime)
        logger.debug(f"Parsed search window: {min_dt} to {max_dt}")
        
        # Validate search window
        if max_dt <= min_dt:
            error_msg = f"Max datetime ({max_dt}) must be after min datetime ({min_dt})"
            logger.error(error_msg)
            return f"Error: {error_msg}"
            
    except ValueError as e:
        logger.error(f"DateTime parsing error: {str(e)}")
        return f"Error parsing datetime: {str(e)}"

    try:
        # Initialize the search tool
        search_tool = CalendarSearchEvents(api_resource=calendar_service)
        
        # Prepare search parameters
        search_params = {
            "calendars_info": calendars_info,
            "min_datetime": min_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "max_datetime": max_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "max_results": max_results,
            "order_by": order_by,
            "query": query,
            "single_events": single_events
        }
        
        logger.debug(f"Search parameters: {search_params}")
        
        # Execute the search
        response = search_tool.invoke(search_params)
        logger.info(f"Search completed, found events in response")
        
        return str(response)
        
    except Exception as e:
        logger.error(f"Failed to search calendar events: {str(e)}", exc_info=True)
        return f"Error searching events: {str(e)}"

@tool(args_schema=UpdateCalEventInput)
@log_function_call
def updateCalEvent(
    event_id: str,
    summary: Optional[str] = None,
    description: Optional[str] = None,
    start_datetime: Optional[datetime] = None,
    end_datetime: Optional[datetime] = None,
    location: Optional[str] = None,
    attendees: Optional[List[str]] = None,
    reminders: Union[None, bool, List[Dict[str, Any]]] = None,
    send_updates: Optional[str] = None,
    timezone: Optional[str] = "Asia/Kolkata"
) -> str:
    """
    Update an existing event in Google Calendar.
    
    This function allows partial updates to calendar events. Only provided
    parameters will be updated; others will remain unchanged.
    
    Args:
        event_id: The unique ID of the event to update
        summary: New event title/summary
        description: New event description
        start_datetime: New start time (datetime object)
        end_datetime: New end time (datetime object)
        location: New event location
        attendees: List of attendee email addresses
        reminders: New reminder configuration
        send_updates: Notification setting - 'all', 'externalOnly', or 'none'
        timezone: Event timezone (default: 'Asia/Kolkata')
    
    Returns:
        JSON string containing the updated event details
        
    Example:
        >>> updateCalEvent(
        ...     event_id="abc123",
        ...     summary="Updated Meeting Title",
        ...     location="Conference Room B",
        ...     send_updates="all"
        ... )
    """
    logger.info(f"Updating calendar event: {event_id}")
    
    # Log which fields are being updated
    update_fields = []
    if summary: update_fields.append("summary")
    if description: update_fields.append("description")
    if start_datetime: update_fields.append("start_datetime")
    if end_datetime: update_fields.append("end_datetime")
    if location: update_fields.append("location")
    if attendees: update_fields.append("attendees")
    if reminders is not None: update_fields.append("reminders")
    
    logger.info(f"Updating fields: {', '.join(update_fields)}")
    
    try:
        # Validate datetime constraints if both are provided
        if start_datetime and end_datetime and end_datetime <= start_datetime:
            error_msg = f"End time ({end_datetime}) must be after start time ({start_datetime})"
            logger.error(error_msg)
            return f"Error: {error_msg}"
        
        # Initialize the update tool
        update_tool = CalendarUpdateEvent(api_resource=calendar_service)

        # Prepare the update payload
        payload = {
            "event_id": event_id,
            "calendar_id": CALENDAR_ID,
            "summary": summary,
            "description": description,
            "start_datetime": start_datetime.strftime("%Y-%m-%d %H:%M:%S") if start_datetime else None,
            "end_datetime": end_datetime.strftime("%Y-%m-%d %H:%M:%S") if end_datetime else None,
            "location": location,
            "attendees": attendees,
            "reminders": reminders,
            "send_updates": send_updates,
            "timezone": timezone
        }

        # Remove None values to avoid updating fields unnecessarily
        payload = {k: v for k, v in payload.items() if v is not None}
        logger.debug(f"Update payload: {payload}")

        # Execute the update
        response = update_tool.invoke(payload)
        logger.info(f"Successfully updated event: {event_id}")
        
        return str(response)
        
    except Exception as e:
        logger.error(f"Failed to update calendar event {event_id}: {str(e)}", exc_info=True)
        return f"Error updating event: {str(e)}"

@tool(args_schema=DeleteCalEventInput)
@log_function_call
def deleteCalEvent(
    event_id: str,
    send_updates: Optional[str] = None
) -> str:
    """
    Delete an event from Google Calendar.
    
    This function permanently removes an event from the calendar.
    Optionally, it can notify attendees of the deletion.
    
    Args:
        event_id: The unique ID of the event to delete
        send_updates: Notification setting for attendees:
                     - 'all': Notify all attendees
                     - 'externalOnly': Notify only external attendees
                     - 'none': Don't send notifications
                     - None: Use default behavior
    
    Returns:
        JSON string containing the API response
        
    Warning:
        This operation is irreversible. Deleted events cannot be recovered.
        
    Example:
        >>> deleteCalEvent(
        ...     event_id="abc123",
        ...     send_updates="all"
        ... )
    """
    logger.info(f"Deleting calendar event: {event_id}")
    logger.warning(f"Event {event_id} will be permanently deleted")
    
    try:
        # Initialize the delete tool
        delete_tool = CalendarDeleteEvent(api_resource=calendar_service)

        # Prepare the delete payload
        payload = {
            "event_id": event_id,
            "calendar_id": CALENDAR_ID,
            "send_updates": send_updates
        }

        # Remove None values
        payload = {k: v for k, v in payload.items() if v is not None}
        logger.debug(f"Delete payload: {payload}")

        # Execute the deletion
        response = delete_tool.invoke(payload)
        logger.info(f"Successfully deleted event: {event_id}")
        
        return str(response)
        
    except Exception as e:
        logger.error(f"Failed to delete calendar event {event_id}: {str(e)}", exc_info=True)
        return f"Error deleting event: {str(e)}"

@tool
@log_function_call
def getCurrentTimeInfo() -> str:
    """
    Get comprehensive current time information for the Asia/Kolkata timezone.
    
    This utility function provides current date, time, day of the week,
    and timezone information. Useful for scheduling and time-based operations.
    
    Returns:
        Formatted string containing current time information
        
    Example Output:
        Current Date: 2023-12-25
        Current Time: 14:30:15
        Day of the Week: Monday
        Timezone: Asia/Kolkata (India)
        
    Note:
        All times are in Asia/Kolkata timezone (IST - Indian Standard Time)
    """
    logger.info("Getting current time information")
    
    try:
        # Get current time in Asia/Kolkata timezone
        tz = pytz.timezone("Asia/Kolkata")
        now = datetime.now(tz)
        
        # Format the time information
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")
        day_of_week = now.strftime("%A")
        timezone_str = "Asia/Kolkata (India)"
        
        # Create formatted response
        time_info = (
            f"Current Date: {date_str}\n"
            f"Current Time: {time_str}\n"
            f"Day of the Week: {day_of_week}\n"
            f"Timezone: {timezone_str}"
        )
        
        logger.info(f"Current time info retrieved: {date_str} {time_str}")
        return time_info
        
    except Exception as e:
        logger.error(f"Failed to get current time info: {str(e)}", exc_info=True)
        return f"Error getting current time: {str(e)}"
