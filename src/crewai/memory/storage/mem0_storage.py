import os
from typing import Any, Dict, List, Optional

from mem0 import MemoryClient
from crewai.memory.storage.interface import Storage


class Mem0Storage(Storage):
    """
    Extends Storage to handle embedding and searching across entities using Mem0.
    """

    def __init__(self, type, crew=None):
        super().__init__()

        if type not in ["user", "agent"]:
            raise ValueError("Invalid type for Mem0Storage. Must be 'user' or 'agent'.")

        self.memory_type = type
        self.memory_config = crew.memory_config

        # User ID is required for user memory type "user" since it's used as a unique identifier for the user.
        user_id = self._get_user_id()
        if type == "user" and not user_id:
            raise ValueError("User ID is required for user memory type")

        # API key in memory config overrides the environment variable
        mem0_api_key = self.memory_config.get("config", {}).get("api_key") or os.getenv(
            "MEM0_API_KEY"
        )
        self.memory = MemoryClient(api_key=mem0_api_key)

    def _sanitize_role(self, role: str) -> str:
        """
        Sanitizes agent roles to ensure valid directory names.
        """
        return role.replace("\n", "").replace(" ", "_").replace("/", "_")

    def save(self, value: Any, metadata: Dict[str, Any]) -> None:
        user_id = self._get_user_id()
        if self.memory_type == "user":
            self.memory.add(value, user_id=user_id, metadata=metadata)
        elif self.memory_type == "agent":
            agent_name = self._get_agent_name()
            self.memory.add(value, agent_id=agent_name, metadata=metadata)

    def search(
        self,
        query: str,
        limit: int = 3,
        filters: Optional[dict] = None,
        score_threshold: float = 0.35,
    ) -> List[Any]:
        params = {"query": query, "limit": limit}
        if self.memory_type == "user":
            user_id = self._get_user_id()
            params["user_id"] = user_id
        elif self.memory_type == "agent":
            agent_name = self._get_agent_name()
            params["agent_id"] = agent_name

        # Discard the filters for now since we create the filters
        # automatically when the crew is created.
        results = self.memory.search(**params)
        return [r for r in results if r["score"] >= score_threshold]

    def _get_user_id(self):
        if self.memory_type == "user":
            return self.memory_config.get("config", {}).get("user_id")
        return None

    def _get_agent_name(self):
        if self.memory_type == "entity":
            return self._sanitize_role(self.memory_config.get("role"))
        return None
