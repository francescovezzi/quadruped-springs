class CollectionBase():
    """Base class to collect a series of elements."""
    def __init__(self):
        self._element_type = None
        self._dict = {}
    
    def _get_el(self, keyword, element_type):
        try:
            return self._dict[keyword]
        except KeyError as e:
            print(f'the {element_type} {keyword} is not implemented yet.')
            
    def get_el(self, keyword):
        return self._get_el(keyword, self._element_type)