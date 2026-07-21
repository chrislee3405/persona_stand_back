from typing import Optional

class CodeService:
    def __init__(self):
        # self.db = db_client
        pass

    async def process_invite_code(self, input_code: str) -> Optional[str]:
        if not input_code.strip():
            return None
        processed_token = ''


        # processing logic


        if input_code == 'asd':
            processed_token = input_code + ' MATCH'
        else:
            processed_token = None

        return processed_token