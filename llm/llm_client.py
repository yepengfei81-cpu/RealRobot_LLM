"""
LLM Client for Real Robot Planning

A lightweight wrapper around OpenAI-compatible APIs (OpenAI, Qwen, etc.)
Handles:
- Synchronous API calls
- Timeout and retry logic
- Error handling
- Response extraction

This module is task-agnostic and can be used for any LLM-based planning task.
"""

import json
import time
from typing import Dict, Any, Optional, Tuple
from pathlib import Path


class LLMClient:
    """
    LLM Client for robot planning tasks.
    
    Supports OpenAI-compatible APIs including:
    - OpenAI GPT models
    - Alibaba Qwen (via DashScope)
    - Other OpenAI-compatible endpoints
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "qwen-plus-2025-12-01",
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        timeout: float = 15.0,
        max_retries: int = 2,
        verbose: bool = True,
    ):
        """
        Initialize LLM Client.
        
        Args:
            api_key: API key for the LLM service
            model: Model name to use
            base_url: Custom API endpoint (None for default OpenAI)
            temperature: Sampling temperature (0.0 for deterministic)
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts on failure
            verbose: Whether to print debug information
        """
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        self.verbose = verbose
        
        # Initialize OpenAI client
        self._client = None
        self._init_client()
    
    def _init_client(self):
        """Initialize the OpenAI client"""
        try:
            from openai import OpenAI
            
            client_kwargs = {
                "api_key": self.api_key,
                "timeout": self.timeout,
            }
            
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
            
            self._client = OpenAI(**client_kwargs)
            
            if self.verbose:
                print(f"[LLM Client] Initialized with model: {self.model}")
                if self.base_url:
                    print(f"[LLM Client] Using custom endpoint: {self.base_url}")
                    
        except ImportError:
            raise ImportError(
                "openai package not found. Install with: pip install openai"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize LLM client: {e}")
    
    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Send a chat request to the LLM.
        
        Args:
            system_prompt: System message defining the assistant's role
            user_prompt: User message with the actual task
            temperature: Override default temperature (optional)
            
        Returns:
            Tuple of (success: bool, response_text: str, error_message: Optional[str])
        """
        if self._client is None:
            return False, "", "LLM client not initialized"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        temp = temperature if temperature is not None else self.temperature
        
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                if self.verbose and attempt > 0:
                    print(f"[LLM Client] Retry attempt {attempt}/{self.max_retries}")
                
                start_time = time.time()
                
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temp,
                )
                
                elapsed = time.time() - start_time
                
                if response.choices and len(response.choices) > 0:
                    content = response.choices[0].message.content
                    
                    if self.verbose:
                        print(f"[LLM Client] Response received in {elapsed:.2f}s")
                    
                    return True, content.strip(), None
                else:
                    last_error = "Empty response from LLM"
                    
            except Exception as e:
                last_error = str(e)
                if self.verbose:
                    print(f"[LLM Client] Error: {last_error}")
                
                # Don't retry on certain errors
                if "authentication" in last_error.lower():
                    break
                if "invalid_api_key" in last_error.lower():
                    break
                
                # Wait before retry
                if attempt < self.max_retries:
                    time.sleep(1.0)
        
        return False, "", last_error
    
    def chat_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
    ) -> Tuple[bool, Optional[Dict], Optional[str]]:
        """
        Send a chat request and parse response as JSON.
        
        Args:
            system_prompt: System message defining the assistant's role
            user_prompt: User message with the actual task
            temperature: Override default temperature (optional)
            
        Returns:
            Tuple of (success: bool, parsed_json: Optional[Dict], error_message: Optional[str])
        """
        success, response_text, error = self.chat(system_prompt, user_prompt, temperature)
        
        if not success:
            return False, None, error
        
        # Try to parse JSON from response
        parsed = self._extract_json(response_text)
        
        if parsed is None:
            return False, None, f"Failed to parse JSON from response: {response_text[:200]}..."
        
        return True, parsed, None
    
    def _extract_json(self, text: str) -> Optional[Dict]:
        """
        Extract JSON from response text.
        
        Handles cases where JSON is wrapped in markdown code blocks.
        """
        if not text:
            return None
        
        # Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # Try to extract from markdown code block
        # Pattern: ```json ... ``` or ``` ... ```
        import re
        
        patterns = [
            r'```json\s*([\s\S]*?)\s*```',
            r'```\s*([\s\S]*?)\s*```',
            r'\{[\s\S]*\}',  # Find any JSON object
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                try:
                    # If pattern captured a group, use it; otherwise use whole match
                    json_str = match if isinstance(match, str) else match[0]
                    return json.loads(json_str.strip())
                except json.JSONDecodeError:
                    continue
        
        return None
    
    @classmethod
    def from_config(cls, config_path: str) -> "LLMClient":
        """
        Create LLMClient from config file.
        
        Args:
            config_path: Path to llm_config.json
            
        Returns:
            Configured LLMClient instance
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path) as f:
            config = json.load(f)
        
        llm_config = config.get("llm", {})
        debug_config = config.get("debug", {})
        
        return cls(
            api_key=llm_config.get("api_key", ""),
            model=llm_config.get("model", "qwen-plus-2025-12-01"),
            base_url=llm_config.get("base_url"),
            temperature=llm_config.get("temperature", 0.0),
            timeout=llm_config.get("timeout", 15.0),
            max_retries=llm_config.get("max_retries", 2),
            verbose=debug_config.get("verbose", True),
        )


# ==================== Convenience Functions ====================

def create_llm_client(config_path: str = None) -> LLMClient:
    """
    Create LLM client with default or specified config.
    
    Args:
        config_path: Path to config file (default: config/llm_config.json)
        
    Returns:
        Configured LLMClient instance
    """
    if config_path is None:
        # Default path relative to this file's location
        default_path = Path(__file__).parent.parent / "config" / "llm_config.json"
        config_path = str(default_path)
    
    return LLMClient.from_config(config_path)


# ==================== Test Code ====================

if __name__ == "__main__":
    """Test the LLM client"""
    
    # Test with config file
    config_path = Path(__file__).parent.parent / "config" / "llm_config.json"
    
    if config_path.exists():
        print(f"Loading config from: {config_path}")
        client = LLMClient.from_config(str(config_path))
        
        # Simple test
        system = "You are a helpful assistant. Respond with valid JSON only."
        user = "Return a JSON object with keys 'status' and 'message'. Set status to 'ok' and message to 'LLM client test successful'."
        
        print("\nSending test request...")
        success, result, error = client.chat_json(system, user)
        
        if success:
            print(f"Success! Response: {result}")
        else:
            print(f"Failed: {error}")
    else:
        print(f"Config file not found: {config_path}")
        print("Please create config/llm_config.json first")