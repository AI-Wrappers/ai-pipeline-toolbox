import re

def resolve_air_urn(air_urn: str) -> str:
    """
    Parses a CivitAI AIR or URN and returns a direct download URL.
    Format example: urn:air:sd1:lora:civitai:12345@67890
    Or just: urn:air:civitai:lora:67890
    
    If it's already a URL, it simply returns it.
    """
    if air_urn.startswith("http://") or air_urn.startswith("https://"):
        return air_urn
        
    # Check if it contains 'civitai'
    if 'civitai' in air_urn.lower():
        # Try to extract the version part which comes after '@'
        match = re.search(r'@(\d+)', air_urn)
        if match:
            version_id = match.group(1)
            return f"https://civitai.com/api/download/models/{version_id}"
            
        # If no '@', but has ':' separated parts, the last part is usually the ID.
        parts = air_urn.split(':')
        last_part = parts[-1]
        if last_part.isdigit():
            # Without version, CivitAI download endpoint still accepts model ID and returns default version
            return f"https://civitai.com/api/download/models/{last_part}"
            
    # Fallback: if we can't parse it reliably, return as is (might fail later)
    return air_urn
