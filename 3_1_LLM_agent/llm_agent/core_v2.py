# llm_agent/core.py

import requests
import json
from typing import List, Dict, Optional
from decouple import config

from .tool_calculator import CalculatorTool
from .tool_websearch import WebSearchTool
from .tool_pdfinfo import PDFInfoTool
from .tool_filesystem import FileSystemTool

class LLMAgent:
    """
    LLM-агент, который планирует и выполняет задачи с помощью инструментов.
    Поддерживает как OpenRouter API, так и локальный Ollama.
    """

    def __init__(self, model: str = "tngtech/deepseek-r1t2-chimera", local: bool = False, 
                 ollama_base_url: str = "http://localhost:11434", ollama_model: str = "qwen3.5:0.8b"):
        """
        Инициализирует агента.
        
        Args:
            model (str): Название модели для OpenRouter.
            local (bool): Если True, использует локальный Ollama вместо OpenRouter.
            ollama_base_url (str): Базовый URL для Ollama API.
            ollama_model (str): Название модели в Ollama.
        """
        self.local = local
        self.ollama_base_url = ollama_base_url
        self.ollama_model = ollama_model
        
        if not self.local:
            self.api_key = config('OPENROUTER_API_KEY')
            self.url = "https://openrouter.ai/api/v1/chat/completions"
            self.model = model
        else:
            self.api_key = None
            self.url = f"{self.ollama_base_url}/v1/chat/completions"
            self.model = ollama_model
        
        # Создаем экземпляры инструментов
        self.tools = {
            "calculator": CalculatorTool(),
            "web_search": WebSearchTool(),
            "pdf_info": PDFInfoTool(),
            "file_system": FileSystemTool(),
        }
        self.conversation_history = []
    
    def _make_api_request(self, payload: Dict, headers: Optional[Dict] = None) -> Dict:
        """
        Универсальный метод для отправки запросов к API.
        Поддерживает как OpenRouter, так и Ollama.
        
        Args:
            payload (Dict): Тело запроса.
            headers (Dict, optional): Заголовки запроса.
            
        Returns:
            Dict: Ответ от API.
        """
        if headers is None:
            headers = {}
        
        if not self.local:
            headers.update({
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            })
        else:
            headers["Content-Type"] = "application/json"
        
        try:
            response = requests.post(self.url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Ошибка при запросе к API: {e}")
    
    def _ask_llm_for_plan(self, query: str) -> List[Dict]:
        """
        Создает план действий, используя LLM.
        Работает как с OpenRouter, так и с Ollama.
        """
        # Системный промпт, который объясняет агенту его роль и формат ответа
        system_prompt = f"""
        You are an assistant that decides if tools are needed.
        Available tools: calculator, web_search, pdf_info, file_system.
        
        Respond ONLY with JSON. Examples:
        If calculator needed: {{"plan": [{{"action": "calculator", "input": "5+3*2"}}]}}
        If web_search needed: {{"plan": [{{"action": "web_search", "input": "погода москва"}}]}}
        If no tool needed: {{"plan": []}}
        """
        
        # Добавляем примеры в запрос
        few_shot = """
        User: Сколько будет 2+2?
        Assistant: {"plan": [{"action": "calculator", "input": "2+2"}]}
        
        User: Привет!
        Assistant: {"plan": []}

        User: Сколько будет 2+2? Когда в последний раз победил Локомотив?
        Assistant: {"plan": [{"action": "calculator", "input": "2+2"},  {"action": "web_search", "input": "локомотив победа"}]}
        """
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "assistant", "content": few_shot},
                {"role": "user", "content": query}
            ]
        }
            
        try:
            # Для Ollama может потребоваться дополнительная настройка
            if self.local:
                # Некоторые модели Ollama могут требовать параметр stream=False
                payload["stream"] = False
            
            response_data = self._make_api_request(payload)
            
            # Извлекаем текстовый ответ от модели
            llm_text = response_data["choices"][0]["message"]["content"]

            # Очищаем ответ от блоков кода Markdown
            import re
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', llm_text, re.DOTALL)
            
            if json_match:
                cleaned_json_text = json_match.group(1)
            else:
                cleaned_json_text = llm_text

            print(f"> Ответ LLM для плана (очищенный): {cleaned_json_text}")
            
            # Пытаемся преобразовать ответ в JSON
            action_plan = json.loads(cleaned_json_text)
            plan = action_plan.get("plan", [])
            return plan
            
        except (json.JSONDecodeError, KeyError, Exception) as e:
            print(f"Произошла ошибка при создании плана: {e}")
            # Пробуем альтернативный подход: извлечь JSON из текста
            try:
                # Ищем JSON в тексте без маркеров
                import re
                json_match = re.search(r'\{.*"plan".*\}', llm_text, re.DOTALL)
                if json_match:
                    action_plan = json.loads(json_match.group())
                    return action_plan.get("plan", [])
            except:
                pass
            return []

    def _generate_final_response(self, user_query: str) -> str:
        """
        Генерирует финальный ответ на основе истории выполнения.
        """
        # Проверяем, есть ли результаты в истории
        tool_results = []
        for msg in self.conversation_history:
            if 'result' in msg['content']:
                tool_results.append(msg['content'])
        
        # Если есть результаты инструментов, используем их напрямую
        if tool_results:
            # Извлекаем наиболее релевантный результат
            # Для поиска - берем первые 3 результата
            combined_results = "\n".join(tool_results)
            
            # Простой промпт для маленькой модели
            prompt = f"""
            Ответь на вопрос, используя информацию из результатов поиска.
            
            Вопрос: {user_query}
            
            Результаты поиска:
            {combined_results[:1500]}  # Ограничиваем длину для маленькой модели
            
            Дай краткий ответ на русском языке.
            """
        else:
            prompt = f"Ответь кратко на вопрос: {user_query}"
        
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        if self.local:
            payload["stream"] = False
        
        try:
            response_data = self._make_api_request(payload)
            final_text = response_data["choices"][0]["message"]["content"]
            return final_text if final_text.strip() else "Не удалось сгенерировать ответ."
        except Exception as e:
            print(f"Ошибка при генерации финального ответа: {e}")
            # Fallback: возвращаем первый найденный результат
            for msg in self.conversation_history:
                if 'result' in msg['content']:
                    return f"Результат поиска: {msg['content'][:500]}"
            return "Извините, не удалось обработать запрос."

    def process_query(self, query: str) -> str:
        """
        Основной метод для обработки запроса пользователя.
        """
        print(f"Агент анализирует ваш запрос... (Режим: {'локальный Ollama' if self.local else 'OpenRouter'})")
        
        # --- Шаг 1: Планирование ---
        plan = self._ask_llm_for_plan(query)

        if not plan:
            print("Инструменты не требуются. Генерирую ответ напрямую.")
            # Генерируем прямой ответ через LLM
            direct_prompt = f"Ответьте на следующий вопрос кратко и информативно: {query}"
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": direct_prompt}]
            }
            if self.local:
                payload["stream"] = False
            try:
                response_data = self._make_api_request(payload)
                return response_data["choices"][0]["message"]["content"]
            except:
                return "Извините, не удалось сгенерировать ответ."

        # --- Шаг 2: Исполнение плана ---
        print(f"План действий: {plan}")
        for step in plan:
            tool_name = step.get('action')
            tool_input = step.get('input')

            if tool_name in self.tools:
                print(f"Выполняется инструмент: '{tool_name}'")
                result = self.tools[tool_name].use(tool_input)
                print(f"Результат: {result}...")
                
                # Добавляем результат в историю
                self.conversation_history.append({
                    'role': 'system',
                    'content': f"Tool {tool_name} result: {result}"
                })
            else:
                error_msg = f"Ошибка: инструмент с именем '{tool_name}' не найден."
                print(error_msg)
                self.conversation_history.append({'role': 'system', 'content': error_msg})
        
        # --- Шаг 3: Генерация финального ответа ---
        print("Составляю финальный ответ...")
        final_response = self._generate_final_response(query)
        return final_response

    def test_ollama_connection(self) -> bool:
        """
        Тестирует соединение с локальным Ollama сервером.
        
        Returns:
            bool: True если соединение успешно, иначе False.
        """
        if not self.local:
            return False
        
        try:
            # Проверяем доступность Ollama API
            test_url = f"{self.ollama_base_url}/v1/models"
            response = requests.get(test_url)
            return response.status_code == 200
        except:
            return False
