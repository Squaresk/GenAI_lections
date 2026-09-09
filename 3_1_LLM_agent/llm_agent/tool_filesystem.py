# llm_agent/tool_filesystem.py

import os
import json
from pathlib import Path
from typing import Optional, Union
import tempfile
import shutil


class FileSystemTool:
    """Инструмент для безопасного выполнения операций с файловой системой в ограниченной директории (песочнице)."""

    name = "file_system"
    description = (
        "Provides safe file operations (read, write, list, delete) within a restricted directory sandbox. "
        "Input should be a JSON string with 'operation' key ('read', 'write', 'list', 'delete') "
        "and operation-specific parameters. All paths are restricted to the sandbox directory."
    )

    def __init__(self, sandbox_dir: Optional[str] = None):
        """
        Initialize the FileSystemTool with a sandbox directory.

        Args:
            sandbox_dir (str, optional): Path to sandbox directory.
                                       If None, uses FILE_SANDBOX_DIR env var or './sandbox' as default.
        """
        # Determine sandbox directory
        if sandbox_dir is None:
            sandbox_dir = os.environ.get('FILE_SANDBOX_DIR', './sandbox')

        # Convert to absolute path and expand user (~)
        self.sandbox_dir = os.path.abspath(os.path.expanduser(sandbox_dir))

        # Create sandbox directory if it doesn't exist
        os.makedirs(self.sandbox_dir, exist_ok=True)

        print(f"> FileSystemTool initialized with sandbox: {self.sandbox_dir}")

    def _validate_path(self, user_path: str) -> str:
        """
        Validate that the user-provided path resolves to a location within the sandbox.

        Args:
            user_path (str): User-provided path (relative to sandbox)

        Returns:
            str: Validated absolute path within sandbox

        Raises:
            ValueError: If path attempts to escape sandbox or is invalid
        """
        if not user_path:
            raise ValueError("Path cannot be empty")

        # Prevent absolute paths and path traversal attempts
        if os.path.isabs(user_path):
            raise ValueError("Absolute paths are not allowed")

        # Normalize and check for path traversal
        normalized_path = os.path.normpath(user_path)
        if normalized_path.startswith("..") or "/.." in normalized_path or "\\.." in normalized_path:
            raise ValueError("Path traversal ('..') is not allowed")

        # Join with sandbox and normalize
        full_path = os.path.normpath(os.path.join(self.sandbox_dir, user_path))

        # Ensure the resolved path is still within sandbox
        if not full_path.startswith(self.sandbox_dir):
            raise ValueError("Access denied: path escapes sandbox")

        return full_path

    def use(self, tool_input: Union[str, dict]) -> str:
        """
        Execute a file system operation based on tool input.

        Args:
            tool_input: Either a JSON string or dict specifying operation and parameters.
                       Expected format: {"operation": "read|write|list|delete", ...operation-specific params...}

        Returns:
            str: Result of operation or error message
        """
        try:
            # Parse input if it's a string
            if isinstance(tool_input, str):
                try:
                    params = json.loads(tool_input)
                except json.JSONDecodeError:
                    return f"Ошибка: не удалось разобрать входные данные как JSON. Ожидалась строка JSON с операцией и параметрами."
            elif isinstance(tool_input, dict):
                params = tool_input
            else:
                return f"Ошибка: неверный тип входных данных. Ожидалась строка JSON или словарь."

            # Extract operation
            operation = params.get("operation")
            if not operation:
                return "Ошибка: не указана операция. Требуется параметр 'operation' с значением 'read', 'write', 'list' или 'delete'."

            # Remove operation from params to avoid passing it to operation methods
            operation_params = {k: v for k, v in params.items() if k != "operation"}

            # Execute operation
            if operation == "read":
                return self._read(**operation_params)
            elif operation == "write":
                return self._write(**operation_params)
            elif operation == "list":
                return self._list(**operation_params)
            elif operation == "delete":
                return self._delete(**operation_params)
            else:
                return f"Ошибка: неизвестная операция '{operation}'. Доступные операции: read, write, list, delete"

        except Exception as e:
            # Return error message consistent with other tools
            return f"Ошибка при выполнении операции: {str(e)}"

    def _read(self, file_path: str) -> str:
        """Read file contents."""
        validated_path = self._validate_path(file_path)

        if not os.path.isfile(validated_path):
            return f"Ошибка: файл '{file_path}' не найден в песочнице."

        with open(validated_path, 'r', encoding='utf-8') as f:
            content = f.read()

        return f"Содержимое файла '{file_path}':\n{content}"

    def _write(self, file_path: str, content: str, mode: str = "w") -> str:
        """Write content to file."""
        validated_path = self._validate_path(file_path)

        # Ensure directory exists
        os.makedirs(os.path.dirname(validated_path) if os.path.dirname(validated_path) else self.sandbox_dir, exist_ok=True)

        with open(validated_path, mode, encoding='utf-8') as f:
            f.write(content)

        return f"Файл '{file_path}' успешно записан ({len(content)} символов)."

    def _list(self, dir_path: str = "") -> str:
        """List directory contents."""
        # For list operation, empty string means root of sandbox
        if dir_path == "":
            validated_path = self.sandbox_dir
        else:
            validated_path = self._validate_path(dir_path)

        if not os.path.isdir(validated_path):
            return f"Ошибка: директория '{dir_path if dir_path else '.'}' не найдена в песочнице."

        items = []
        for item in sorted(os.listdir(validated_path)):
            item_path = os.path.join(validated_path, item)
            if os.path.isdir(item_path):
                items.append(f"📁 {item}/")
            else:
                size = os.path.getsize(item_path)
                items.append(f"📄 {item} ({size} байт)")

        if not items:
            return f"Директория '{dir_path if dir_path else '.'}' пуста."

        return f"Содержимое директории '{dir_path if dir_path else '.'}':\n" + "\n".join(items)

    def _delete(self, file_path: str) -> str:
        """Delete file or directory."""
        validated_path = self._validate_path(file_path)

        if not os.path.exists(validated_path):
            return f"Ошибка: цель '{file_path}' не найдена в песочнице."

        if os.path.isfile(validated_path):
            os.remove(validated_path)
            return f"Файл '{file_path}' успешно удален."
        elif os.path.isdir(validated_path):
            # Remove directory and all contents
            shutil.rmtree(validated_path)
            return f"Директория '{file_path}' и все ее содержимое успешно удалены."
        else:
            return f"Ошибка: неизвестный тип цели '{file_path}'."