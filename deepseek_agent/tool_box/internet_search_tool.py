import os
from googleapiclient.discovery import build
from tool_box.base_tool import BaseTool
from typing import List, Dict, Tuple


class InternetSearchTool(BaseTool):
    name = "internet_search"
    def __init__(self):
        # Load keys from environment variables
        api_key = os.getenv("GOOGLE_API_KEY")
        cse_id = os.getenv("GOOGLE_CSE_ID")
        if not api_key or not cse_id:
            raise ValueError("GOOGLE_API_KEY and GOOGLE_CSE_ID must be set in the environment.")
            
        self.description_text = "Returns a list of relevant document snippets for a textual query retrieved from the internet using Google. Use this tool in combination with the browser tool, use google to find links and then open the link with the browser tool."
        self.parameter_definitions = {
            "query": {
                "description": "Query to search the internet with",
                "type": 'str',
                "required": True
            }
        }
        self._parameter_example_usage = {
            "query": "latest news on artificial intelligence"
        }

        self.api_key = api_key
        self.cse_id = cse_id
        self.num = 5

    def description(self) -> Dict[str, str]:
        return {
            "name": self.name,
            "description": self.description_text,
            "parameter_definitions": self.parameter_definitions
        }

    def run(self, query: str, *args, **kwargs) -> List[Dict[str, str]]:
        service = build("customsearch", "v1", developerKey=self.api_key)
        results = service.cse().list(q=query, cx=self.cse_id, num=self.num).execute()

        search_results = []
        for item in results.get("items", []):
            search_results.append({
                "title": item["title"],
                "snippet": item["snippet"],
                "link": item["link"]
            })

        return search_results

    def __call__(self, query: str) -> List[Dict[str, str]]:
        return self.run(query)