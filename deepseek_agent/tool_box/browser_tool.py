from tool_box.base_tool import BaseTool
from typing import List, Dict
import trafilatura
import signal

def page_result(text: str, cursor: int, max_length: int) -> str:
    """Page through `text` and return a substring of `max_length` characters starting from `cursor`."""
    return text[cursor : cursor + max_length]

# Define a custom exception for handling timeouts
class TimeoutException(Exception):
    pass

# Define the signal handler
def signal_handler(signum, frame):
    raise TimeoutException("The operation timed out.")

# Function to fetch and extract text with a timeout
def fetch_and_extract_text(url, timeout=10):
    # Register the signal handler and set the alarm
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(timeout)

    try:
        downloaded = trafilatura.fetch_url(url)
        extracted_text = trafilatura.extract(downloaded)
        if not extracted_text:
            return "Could not extract content from the provided URL."
    except TimeoutException as te:
        return str(te)
    except Exception as e:
        return f"An error occurred: {str(e)}"
    finally:
        # Disable the alarm
        signal.alarm(0)

    return extracted_text

class BrowserTool(BaseTool):
    name = "browser"
    max_result_length = 2000

    def __init__(self):
        super().__init__()  # Initialize base class
        # Define the parameters and description for this tool
        self.description_text = "A simple text-based browser tool that extracts main content from web pages. You can use this tool in combination with internet_search to open links."
        self.parameter_definitions = {
            "url": {
                "description": "The URL of the web page to extract content from",
                "type": 'str',
                "required": True
            },
            "cursor": {
                "description": "The position to start reading from in the extracted content",
                "type": 'int',
                "required": False
            }
        }
        self._parameter_example_usage = {
            "url": "https://en.wikipedia.org/wiki/Artificial_intelligence",
            "cursor": 0
        }

    def description(self) -> Dict[str, str]:
        # Provide a description of this tool, including its parameters
        return {
            "name": self.name,
            "description": self.description_text,
            "parameter_definitions": self.parameter_definitions
        }

    def run(self, url: str, cursor: int = 0, *args, **kwargs) -> str:
        # Use Trafilatura to fetch and extract main content from the given URL

       
        if type(cursor) != int:
            try:
                cursor = int(cursor)
            except ValueError:
                cursor = 0
        elif cursor < 0:
            cursor = 0

        # downloaded = trafilatura.fetch_url(url)
        extracted_text = fetch_and_extract_text(url)

        if not extracted_text:
            return "Could not extract content from the provided URL."

        if len(extracted_text) > self.max_result_length:
            # either use the cursor or a summary llm
            page_contents = page_result(extracted_text, cursor, self.max_result_length)
            # check if the page was truncated or if it is the end of the page
            if len(page_contents) < self.max_result_length:
                page_contents += f"\nEND OF PAGE. DON'T USE CURSOR."
            else:
                page_contents += f"\nPAGE WAS TRUNCATED. ONLY IF NECESSARY: TO CONTINUE READING, USE CURSOR={cursor+len(page_contents)}."
        else:
            page_contents = extracted_text + f"\nEND OF PAGE. DON'T USE CURSOR."
        
        return page_contents

    def __call__(self, url: str) -> str:
        # Make this class callable, allowing it to be used like a function
        return self.run(url)
