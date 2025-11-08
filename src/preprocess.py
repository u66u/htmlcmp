import html, re
import os
import yaml
from typing import List, Dict, Set, Tuple, Iterable, Optional
from dataclasses import dataclass, field
import requests
from urllib.parse import urlparse
import trafilatura

@dataclass
class CanonicalizeConfig:
    # Filter toggles
    filter_short_meta: bool = True
    filter_read_time: bool = True
    filter_date_lines: bool = True
    filter_digit_heavy: bool = True
    normalize_handles: bool = True
    normalize_urls: bool = True
    
    # Thresholds
    short_line_threshold: int = 20
    digit_heavy_threshold: float = 0.3  # Min ratio of letters to total chars
    
    # Replacement tokens
    handle_token: str = "[HANDLE]"
    url_token: str = "[URL]"
    
    # Sentence punctuation for short line exceptions
    sentence_punctuation: tuple = ('.', '!', '?', ':', ';')
    
    @classmethod
    def load(cls, yaml_path: Optional[str] = None, env_prefix: str = "CANON_", **overrides) -> 'CanonicalizeConfig':
        config_dict = {}
        
        if yaml_path and os.path.exists(yaml_path):
            with open(yaml_path, 'r') as f:
                yaml_config = yaml.safe_load(f) or {}
                config_dict.update({k: v for k, v in yaml_config.items() if hasattr(cls, k)})
        
        env_mappings = {
            f"{env_prefix}FILTER_SHORT_META": ("filter_short_meta", lambda x: x.lower() == 'true'),
            f"{env_prefix}FILTER_READ_TIME": ("filter_read_time", lambda x: x.lower() == 'true'),
            f"{env_prefix}FILTER_DATE_LINES": ("filter_date_lines", lambda x: x.lower() == 'true'),
            f"{env_prefix}FILTER_DIGIT_HEAVY": ("filter_digit_heavy", lambda x: x.lower() == 'true'),
            f"{env_prefix}NORMALIZE_HANDLES": ("normalize_handles", lambda x: x.lower() == 'true'),
            f"{env_prefix}NORMALIZE_URLS": ("normalize_urls", lambda x: x.lower() == 'true'),
            f"{env_prefix}SHORT_LINE_THRESHOLD": ("short_line_threshold", int),
            f"{env_prefix}DIGIT_HEAVY_THRESHOLD": ("digit_heavy_threshold", float),
            f"{env_prefix}HANDLE_TOKEN": ("handle_token", str),
            f"{env_prefix}URL_TOKEN": ("url_token", str),
        }
        
        for env_var, (field_name, converter) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                config_dict[field_name] = converter(value)
        
        config_dict.update(overrides)
        
        return cls(**config_dict)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'CanonicalizeConfig':
        if not os.path.exists(yaml_path):
            return cls()
        
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f) or {}
        
        return cls(**{k: v for k, v in config_dict.items() if hasattr(cls, k)})
    
    @classmethod
    def from_env(cls, prefix: str = "CANON_") -> 'CanonicalizeConfig':
        config_dict = {}
        
        env_mappings = {
            f"{prefix}FILTER_SHORT_META": ("filter_short_meta", lambda x: x.lower() == 'true'),
            f"{prefix}FILTER_READ_TIME": ("filter_read_time", lambda x: x.lower() == 'true'),
            f"{prefix}FILTER_DATE_LINES": ("filter_date_lines", lambda x: x.lower() == 'true'),
            f"{prefix}FILTER_DIGIT_HEAVY": ("filter_digit_heavy", lambda x: x.lower() == 'true'),
            f"{prefix}NORMALIZE_HANDLES": ("normalize_handles", lambda x: x.lower() == 'true'),
            f"{prefix}NORMALIZE_URLS": ("normalize_urls", lambda x: x.lower() == 'true'),
            f"{prefix}SHORT_LINE_THRESHOLD": ("short_line_threshold", int),
            f"{prefix}DIGIT_HEAVY_THRESHOLD": ("digit_heavy_threshold", float),
            f"{prefix}HANDLE_TOKEN": ("handle_token", str),
            f"{prefix}URL_TOKEN": ("url_token", str),
        }
        
        for env_var, (field_name, converter) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                config_dict[field_name] = converter(value)
        
        return cls(**config_dict)


def fetch(url: str, timeout=15) -> str | None:
    r = requests.get(url, timeout=timeout)
    if r.status_code != 200:
        return None
    return r.text

def extract_article(html_str: str) -> str:
    res = trafilatura.extract(html_str, include_comments=False, include_tables=False)
    return res or ""

def canonicalize(content: str, config: Optional[CanonicalizeConfig] = None) -> str:
    if config is None:
        config = CanonicalizeConfig()
    
    lines = content.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        
        if not line:
            continue
        
        if config.filter_short_meta:
            if len(line) < config.short_line_threshold:
                if not line.endswith(config.sentence_punctuation):
                    continue
        
        if config.filter_read_time:
            if re.search(r'\d+\s*min(ute)?s?\s+read', line, re.IGNORECASE):
                continue
        
        if config.filter_date_lines:
            date_patterns = [
                r'^\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\s*$',
                r'^\s*\d{4}[/-]\d{1,2}[/-]\d{1,2}\s*$',
                r'^\s*(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}\s*$',
                r'^\s*\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{4}\s*$',
            ]
            if any(re.match(pattern, line, re.IGNORECASE) for pattern in date_patterns):
                continue
        
        if config.filter_digit_heavy:
            letter_count = sum(c.isalpha() for c in line)
            if len(line) > 0 and letter_count / len(line) < config.digit_heavy_threshold:
                continue
        
        if config.normalize_handles:
            line = re.sub(r'@\w+', config.handle_token, line)
        
        if config.normalize_urls:
            line = re.sub(r'https?://\S+', config.url_token, line)
        
        cleaned_lines.append(line)
    
    s = ' '.join(cleaned_lines)
    s = html.unescape(s)
    s = s.lower()
    s = re.sub(r'\s+', ' ', s)
    s = s.strip()
    
    return s


def tokenize(s: str) -> List[str]:
    return re.findall(r'\w+', s)


def k_gram(tokens: List[str], k: int) -> List[str]:
    if len(tokens) < k: return []
    return [' '.join(tokens[i:i+k]) for i in range(len(tokens)-k+1)]