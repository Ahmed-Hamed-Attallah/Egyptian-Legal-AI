"""
Utility functions for date parsing and formatting in legal contexts.
Handles various date formats found in Egyptian legal documents.
"""
from datetime import datetime, timezone
from typing import Optional

from src.config.constants import LEGAL_DATE_FORMATS


def parse_date(date_string: str, formats: Optional[list[str]] = None) -> Optional[datetime]:
    """
    Attempt to parse a date string using multiple formats.
    
    Args:
        date_string: The date string to parse
        formats: List of datetime format strings to try (defaults to LEGAL_DATE_FORMATS)
    
    Returns:
        Parsed datetime object or None if parsing fails
    
    Example:
        parse_date("2023-10-31") -> datetime(2023, 10, 31)
        parse_date("31/10/2023") -> datetime(2023, 10, 31)
        parse_date("invalid") -> None
    """
    if not date_string or not date_string.strip():
        return None
    
    formats = formats or LEGAL_DATE_FORMATS
    
    for fmt in formats:
        try:
            return datetime.strptime(date_string.strip(), fmt)
        except (ValueError, TypeError):
            continue
    
    return None


def parse_arabic_date(date_string: str) -> Optional[datetime]:
    """
    Parse dates with Arabic month names.
    
    Supports formats like:
    - "31 أكتوبر 2023"
    - "1 يناير 2024"
    
    Args:
        date_string: Date string with Arabic month names
    
    Returns:
        Parsed datetime object or None if parsing fails
    """
    arabic_months = {
        "يناير": "01", "فبراير": "02", "مارس": "03", "أبريل": "04",
        "مايو": "05", "يونيو": "06", "يوليو": "07", "أغسطس": "08",
        "سبتمبر": "09", "أكتوبر": "10", "نوفمبر": "11", "ديسمبر": "12",
    }
    
    # Try to find and replace Arabic month names with numbers
    for arabic_month, month_num in arabic_months.items():
        if arabic_month in date_string:
            # Replace Arabic month with number and try standard parsing
            normalized = date_string.replace(arabic_month, month_num)
            # Try format: "31 10 2023"
            result = parse_date(normalized, ["%d %m %Y", "%d %m %y"])
            if result:
                return result
    
    return None


def format_date_for_display(
    date_obj: datetime, 
    language: str = "ar",
    include_time: bool = False
) -> str:
    """
    Format a datetime object for user display.
    
    Args:
        date_obj: The datetime object to format
        language: "ar" for Arabic, "en" for English
        include_time: Whether to include time in the output
    
    Returns:
        Formatted date string
    
    Example:
        format_date_for_display(datetime(2023, 10, 31), "ar") -> "31 أكتوبر 2023"
        format_date_for_display(datetime(2023, 10, 31), "en") -> "31 October 2023"
    """
    if language == "ar":
        arabic_months = [
            "يناير", "فبراير", "مارس", "أبريل", "مايو", "يونيو",
            "يوليو", "أغسطس", "سبتمبر", "أكتوبر", "نوفمبر", "ديسمبر"
        ]
        month_name = arabic_months[date_obj.month - 1]
        formatted = f"{date_obj.day} {month_name} {date_obj.year}"
        
        if include_time:
            formatted += f" - {date_obj.strftime('%H:%M')}"
        
        return formatted
    else:
        if include_time:
            return date_obj.strftime("%d %B %Y - %H:%M")
        return date_obj.strftime("%d %B %Y")


def get_current_utc_timestamp() -> datetime:
    """
    Get the current UTC timestamp.
    Used for database record creation and audit trails.
    
    Returns:
        Current datetime in UTC timezone
    """
    return datetime.now(timezone.utc)


def format_timestamp_for_db(dt: datetime) -> str:
    """
    Format a datetime object for database storage (ISO 8601).
    
    Args:
        dt: The datetime object to format
    
    Returns:
        ISO 8601 formatted string
    
    Example:
        format_timestamp_for_db(datetime(2023, 10, 31, 15, 30)) -> "2023-10-31T15:30:00"
    """
    return dt.isoformat()


def parse_timestamp_from_db(timestamp_str: str) -> Optional[datetime]:
    """
    Parse an ISO 8601 timestamp string from the database.
    
    Args:
        timestamp_str: ISO 8601 formatted string
    
    Returns:
        Parsed datetime object or None if parsing fails
    """
    try:
        return datetime.fromisoformat(timestamp_str)
    except (ValueError, TypeError):
        return None