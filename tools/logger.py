#!/usr/bin/env python3
"""
SEC-FLOYD Logger Module

This module provides logging functionality to track API calls, responses, and agent actions.
Each run creates a timestamped log file in the logs directory.
"""

import os
import json
import logging
import datetime
import time
from functools import wraps

# Ensure logs directory exists
logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
os.makedirs(logs_dir, exist_ok=True)

# Create timestamped filename for this session
timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')
log_file = os.path.join(logs_dir, f'sec_floyd_{timestamp}.log')

# Simple formatter that uses local time with AM/PM
class SimpleFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        # Get current local time using time.localtime()
        ct = time.localtime(record.created)
        if datefmt:
            s = time.strftime(datefmt, ct)
        else:
            # Format with 12-hour clock and AM/PM
            s = time.strftime("%Y-%m-%d %I:%M:%S %p", ct)
        return s

# Configure root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Create formatter with simple time format
formatter = SimpleFormatter('%(asctime)s | %(levelname)8s | %(name)25s | %(message)s')

# Clear any existing handlers (to avoid duplicate logs)
if root_logger.handlers:
    root_logger.handlers.clear()

# Create and add handlers
file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(formatter)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

root_logger.addHandler(file_handler)
root_logger.addHandler(console_handler)

def get_logger(name):
    """Get a logger instance with the given name."""
    return logging.getLogger(name)

# Create a main logger instance
logger = get_logger('sec_floyd')

def format_api_data(data):
    """Format API data for logging, truncating long values."""
    if not data:
        return "None"
    
    try:
        if isinstance(data, str) and len(data) > 1000:
            return f"{data[:500]}... [truncated, total length: {len(data)}]"
        elif isinstance(data, dict):
            # Create a copy to avoid modifying the original
            formatted = {}
            for k, v in data.items():
                # Skip logging API keys
                if 'api_key' in k.lower() or 'apikey' in k.lower():
                    formatted[k] = '[REDACTED]'
                elif isinstance(v, str) and len(v) > 1000:
                    formatted[k] = f"{v[:500]}... [truncated, total length: {len(v)}]"
                elif isinstance(v, (dict, list)):
                    formatted[k] = format_api_data(v)
                else:
                    formatted[k] = v
            return formatted
        elif isinstance(data, list) and len(data) > 20:
            return f"{data[:10]}... [truncated, total items: {len(data)}]"
        else:
            return data
    except Exception as e:
        return f"Error formatting data: {str(e)}"

def log_api_call(func):
    """Decorator to log API calls and responses."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        module = func.__module__
        logger = get_logger(f'{module}.{func_name}')
        
        # Log the call
        safe_kwargs = {k: ('[REDACTED]' if 'api_key' in k.lower() or 'apikey' in k.lower() else v) 
                      for k, v in kwargs.items()}
        
        logger.info(f"API CALL | Function: {func_name} | Args: {format_api_data(args[1:] if len(args) > 1 else None)} | Kwargs: {format_api_data(safe_kwargs)}")
        
        try:
            # Execute the function
            result = func(*args, **kwargs)
            
            # Log the response
            logger.info(f"API RESPONSE | Function: {func_name} | Status: {result.get('status', 'unknown')} | Result: {format_api_data(result)}")
            
            return result
        except Exception as e:
            logger.error(f"API ERROR | Function: {func_name} | Error: {str(e)}", exc_info=True)
            raise
    
    return wrapper

def log_user_interaction(query, response=None):
    """Log user queries and agent responses."""
    logger = get_logger('user.interaction')
    if response is None:
        logger.info(f"USER QUERY | {query}")
    else:
        logger.info(f"AGENT RESPONSE | Query: {query[:100]}... | Response: {format_api_data(response)}") 