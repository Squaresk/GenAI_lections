import pytest
import os
import tempfile
import shutil
from llm_agent.tool_filesystem import FileSystemTool


def test_filesystem_initialization():
    """Test that FileSystemTool initializes correctly with default and custom sandbox."""
    # Test default sandbox (should use env var or ./sandbox)
    tool = FileSystemTool()
    assert tool.name == "file_system"
    assert "sandbox" in tool.sandbox_dir.lower() or tool.sandbox_dir.endswith(os.path.sep + "sandbox")

    # Test that sandbox directory is created
    assert os.path.exists(tool.sandbox_dir)

    # Cleanup
    if os.path.exists(tool.sandbox_dir) and tool.sandbox_dir.endswith("sandbox"):
        shutil.rmtree(tool.sandbox_dir, ignore_errors=True)


def test_filesystem_sandbox_security():
    """Test that FileSystemTool properly restricts operations to sandbox."""
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        sandbox_dir = os.path.join(temp_dir, "test_sandbox")
        tool = FileSystemTool(sandbox_dir)

        # Test successful operation within sandbox
        result = tool.use('{"operation": "write", "file_path": "test.txt", "content": "Hello World"}')
        assert "успешно записан" in result
        assert os.path.exists(os.path.join(sandbox_dir, "test.txt"))

        # Test reading the file
        result = tool.use('{"operation": "read", "file_path": "test.txt"}')
        assert "Содержимое файла 'test.txt'" in result
        assert "Hello World" in result

        # Test listing files
        result = tool.use('{"operation": "list", "dir_path": ""}')
        assert "test.txt" in result

        # Test deleting file
        result = tool.use('{"operation": "delete", "file_path": "test.txt"}')
        assert "успешно удален" in result
        assert not os.path.exists(os.path.join(sandbox_dir, "test.txt"))

        # Test directory traversal protection
        result = tool.use('{"operation": "read", "file_path": "../../etc/passwd"}')
        assert "Ошибка" in result
        assert "Path traversal" in result or "Access denied" in result

        # Test absolute path rejection
        result = tool.use('{"operation": "read", "file_path": "/etc/passwd"}')
        assert "Ошибка" in result
        assert "Absolute paths are not allowed" in result


def test_filesystem_operations():
    """Test basic file system operations."""
    with tempfile.TemporaryDirectory() as temp_dir:
        sandbox_dir = os.path.join(temp_dir, "test_sandbox")
        tool = FileSystemTool(sandbox_dir)

        # Test write operation
        result = tool.use('{"operation": "write", "file_path": "document.txt", "content": "This is a test document."}')
        assert "успешно записан" in result
        assert os.path.exists(os.path.join(sandbox_dir, "document.txt"))

        # Test read operation
        result = tool.use('{"operation": "read", "file_path": "document.txt"}')
        assert "Содержимое файла 'document.txt'" in result
        assert "This is a test document." in result

        # Test list operation
        result = tool.use('{"operation": "list", "dir_path": ""}')
        assert "document.txt" in result

        # Test write with append mode
        result = tool.use('{"operation": "write", "file_path": "document.txt", "content": "\\nAdditional line.", "mode": "a"}')
        assert "успешно записан" in result

        # Test read after append
        result = tool.use('{"operation": "read", "file_path": "document.txt"}')
        assert "This is a test document." in result
        assert "Additional line." in result

        # Test directory creation and listing
        result = tool.use('{"operation": "write", "file_path": "subdir/nested.txt", "content": "Nested file"}')
        assert "успешно записан" in result

        result = tool.use('{"operation": "list", "dir_path": "subdir"}')
        assert "nested.txt" in result

        # Test delete directory
        result = tool.use('{"operation": "delete", "file_path": "subdir"}')
        assert "успешно удалены" in result
        assert not os.path.exists(os.path.join(sandbox_dir, "subdir"))


def test_filesystem_error_handling():
    """Test error handling for various edge cases."""
    with tempfile.TemporaryDirectory() as temp_dir:
        sandbox_dir = os.path.join(temp_dir, "test_sandbox")
        tool = FileSystemTool(sandbox_dir)

        # Test reading non-existent file
        result = tool.use('{"operation": "read", "file_path": "nonexistent.txt"}')
        assert "Ошибка" in result
        assert "не найден" in result

        # Test deleting non-existent file
        result = tool.use('{"operation": "delete", "file_path": "nonexistent.txt"}')
        assert "Ошибка" in result
        assert "не найден" in result

        # Test listing non-existent directory
        result = tool.use('{"operation": "list", "dir_path": "nonexistent"}')
        assert "Ошибка" in result
        assert "не найден" in result

        # Test invalid operation
        result = tool.use('{"operation": "invalid_op", "file_path": "test.txt"}')
        assert "Ошибка: неизвестная операция" in result

        # Test missing operation parameter
        result = tool.use('{"file_path": "test.txt"}')
        assert "Ошибка: не указана операция" in result

        # Test invalid JSON
        result = tool.use('{"operation": "read", "file_path": "test.txt"')
        assert "Ошибка: не удалось разобрать входные данные как JSON" in result


if __name__ == "__main__":
    # Run tests manually if needed
    test_filesystem_initialization()
    print("✓ Initialization test passed")

    test_filesystem_sandbox_security()
    print("✓ Sandbox security test passed")

    test_filesystem_operations()
    print("✓ Operations test passed")

    test_filesystem_error_handling()
    print("✓ Error handling test passed")

    print("All tests passed!")