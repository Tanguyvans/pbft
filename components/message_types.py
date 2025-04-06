from enum import Enum

class MessageType(str, Enum):
    REQUEST = 'request'
    PRE_PREPARE = 'pre-prepare'
    PREPARE = 'prepare'
    COMMIT = 'commit'
    VIEW_CHANGE = 'view-change'
    NEW_VIEW = 'new-view'
    HEARTBEAT = 'heartbeat'
    BLOCK_SYNC = 'block-sync'
    STATE_SYNC = 'state-sync'
    VIEW_SYNC = 'view-sync'
    NODE_JOIN = 'node-join'
    MODEL_REQUEST = 'model-request'
    VALIDATION_RESULT = 'validation-result'
    VALIDATION_FAILED = 'validation-failed'

    def __str__(self):
        return self.value