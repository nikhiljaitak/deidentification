from enum import Enum

class AppEnum(Enum):
    SIMILARITY_THRESHOLD=0.65
    BLANK_STRING=''
    REGEX_ONLY_DIGITS='[^0-9]'
    REGEX_MEMBER_ID=['\s?[wW]\d{9}\s?', 'w(\d+){9}\s?','w(\d+){4}\s?(\d){5}\s?','w(\d+){5}\s(\d){4}\s?','w\s?(\d+){9}\s?','mbr#*\s*(\d+){7}\s*','mbr*\s*(\d+){7}\s*','-\s*(\d+){7}\s*','\s*(\d+){7}\s*-','member id=([0-9]){9}\s?','\s\d{7}\s','\s\d{8}\s','\s\d{9}\s','\s\d{10}\s']
    FUNCTION_CURRENT_PLACEHOLDER="The current function name is:"
    REGEX_EMAIL='[a-zA-Z0-9._-]+@[a-zA-Z0-9._-]+\.[a-zA-Z0-9_-]+'
    REGEX_SSN='\d{3}-\d{2}-\d{4}'
    RUNNING_STATUS_MESSAGE="Application in Running Status"
