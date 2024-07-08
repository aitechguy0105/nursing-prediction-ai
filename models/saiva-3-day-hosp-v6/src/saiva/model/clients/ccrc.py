import sys

from .base import Base

class Ccrc(Base):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.facilities = "2"
