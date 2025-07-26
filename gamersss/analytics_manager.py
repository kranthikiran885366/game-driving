import json
import time
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

class EventType(Enum):
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    GAME_MODE_START = "game_mode_start"
    GAME_MODE_END = "game_mode_end"
    VEHICLE_SELECT = "vehicle_select"
    UPGRADE_PURCHASE = "upgrade_purchase"
    ACHIEVEMENT_UNLOCK = "achievement_unlock"
    CHALLENGE_COMPLETE = "challenge_complete"
    CRASH = "crash"
    GESTURE_DETECT = "gesture_detect"
    ERROR = "error"
    PERFORMANCE = "performance"

@dataclass
class GameEvent:
    type: EventType
    timestamp: datetime
    data: Dict
    session_id: str

@dataclass
class SessionData:
    start_time: datetime
    end_time: Optional[datetime]
    events: List[GameEvent]
    game_modes_played: Dict[str, int]
    total_distance: float
    total_crashes: int
    total_coins_earned: int
    total_coins_spent: int
    performance_metrics: Dict[str, float]

class AnalyticsManager:
    def __init__(self):
        self.current_session: Optional[SessionData] = None
        self.sessions: List[SessionData] = []
        self._load_data()
    
    def _load_data(self):
        try:
            with open('analytics_data.json', 'r') as f:
                data = json.load(f)
                # Load historical sessions
                for session_data in data.get('sessions', []):
                    session = SessionData(
                        start_time=datetime.fromisoformat(session_data['start_time']),
                        end_time=datetime.fromisoformat(session_data['end_time']) if session_data.get('end_time') else None,
                        events=[],
                        game_modes_played=session_data.get('game_modes_played', {}),
                        total_distance=session_data.get('total_distance', 0),
                        total_crashes=session_data.get('total_crashes', 0),
                        total_coins_earned=session_data.get('total_coins_earned', 0),
                        total_coins_spent=session_data.get('total_coins_spent', 0),
                        performance_metrics=session_data.get('performance_metrics', {})
                    )
                    
                    # Load events
                    for event_data in session_data.get('events', []):
                        event = GameEvent(
                            type=EventType(event_data['type']),
                            timestamp=datetime.fromisoformat(event_data['timestamp']),
                            data=event_data['data'],
                            session_id=event_data['session_id']
                        )
                        session.events.append(event)
                    
                    self.sessions.append(session)
        except FileNotFoundError:
            self._save_data()
    
    def _save_data(self):
        data = {
            'sessions': [
                {
                    'start_time': session.start_time.isoformat(),
                    'end_time': session.end_time.isoformat() if session.end_time else None,
                    'events': [
                        {
                            'type': event.type.value,
                            'timestamp': event.timestamp.isoformat(),
                            'data': event.data,
                            'session_id': event.session_id
                        }
                        for event in session.events
                    ],
                    'game_modes_played': session.game_modes_played,
                    'total_distance': session.total_distance,
                    'total_crashes': session.total_crashes,
                    'total_coins_earned': session.total_coins_earned,
                    'total_coins_spent': session.total_coins_spent,
                    'performance_metrics': session.performance_metrics
                }
                for session in self.sessions
            ]
        }
        
        with open('analytics_data.json', 'w') as f:
            json.dump(data, f, indent=4)
    
    def start_session(self):
        """Start a new analytics session"""
        self.current_session = SessionData(
            start_time=datetime.now(),
            end_time=None,
            events=[],
            game_modes_played={},
            total_distance=0,
            total_crashes=0,
            total_coins_earned=0,
            total_coins_spent=0,
            performance_metrics={}
        )
        self._log_event(EventType.SESSION_START, {})
    
    def end_session(self):
        """End the current analytics session"""
        if self.current_session:
            self.current_session.end_time = datetime.now()
            self._log_event(EventType.SESSION_END, {})
            self.sessions.append(self.current_session)
            self.current_session = None
            self._save_data()
    
    def _log_event(self, event_type: EventType, data: Dict):
        """Log an event to the current session"""
        if self.current_session:
            event = GameEvent(
                type=event_type,
                timestamp=datetime.now(),
                data=data,
                session_id=str(self.current_session.start_time.timestamp())
            )
            self.current_session.events.append(event)
    
    def log_game_mode_start(self, mode: str):
        """Log the start of a game mode"""
        if self.current_session:
            self.current_session.game_modes_played[mode] = self.current_session.game_modes_played.get(mode, 0) + 1
            self._log_event(EventType.GAME_MODE_START, {'mode': mode})
    
    def log_game_mode_end(self, mode: str, score: int, distance: float):
        """Log the end of a game mode"""
        if self.current_session:
            self.current_session.total_distance += distance
            self._log_event(EventType.GAME_MODE_END, {
                'mode': mode,
                'score': score,
                'distance': distance
            })
    
    def log_vehicle_select(self, vehicle: str):
        """Log vehicle selection"""
        self._log_event(EventType.VEHICLE_SELECT, {'vehicle': vehicle})
    
    def log_upgrade_purchase(self, upgrade_type: str, cost: int):
        """Log an upgrade purchase"""
        if self.current_session:
            self.current_session.total_coins_spent += cost
            self._log_event(EventType.UPGRADE_PURCHASE, {
                'upgrade_type': upgrade_type,
                'cost': cost
            })
    
    def log_achievement_unlock(self, achievement_id: str, reward: int):
        """Log an achievement unlock"""
        if self.current_session:
            self.current_session.total_coins_earned += reward
            self._log_event(EventType.ACHIEVEMENT_UNLOCK, {
                'achievement_id': achievement_id,
                'reward': reward
            })
    
    def log_challenge_complete(self, challenge_id: str, reward: int):
        """Log challenge completion"""
        if self.current_session:
            self.current_session.total_coins_earned += reward
            self._log_event(EventType.CHALLENGE_COMPLETE, {
                'challenge_id': challenge_id,
                'reward': reward
            })
    
    def log_crash(self, speed: float, damage: float):
        """Log a vehicle crash"""
        if self.current_session:
            self.current_session.total_crashes += 1
            self._log_event(EventType.CRASH, {
                'speed': speed,
                'damage': damage
            })
    
    def log_gesture_detect(self, gesture: str, confidence: float):
        """Log gesture detection"""
        self._log_event(EventType.GESTURE_DETECT, {
            'gesture': gesture,
            'confidence': confidence
        })
    
    def log_error(self, error_type: str, error_message: str):
        """Log an error"""
        self._log_event(EventType.ERROR, {
            'error_type': error_type,
            'error_message': error_message
        })
    
    def log_performance(self, fps: float, frame_time: float, memory_usage: float):
        """Log performance metrics"""
        if self.current_session:
            self.current_session.performance_metrics.update({
                'fps': fps,
                'frame_time': frame_time,
                'memory_usage': memory_usage
            })
            self._log_event(EventType.PERFORMANCE, {
                'fps': fps,
                'frame_time': frame_time,
                'memory_usage': memory_usage
            })
    
    def get_session_duration(self) -> Optional[float]:
        """Get the duration of the current session in seconds"""
        if self.current_session:
            return (datetime.now() - self.current_session.start_time).total_seconds()
        return None
    
    def get_average_session_duration(self) -> float:
        """Calculate average session duration across all sessions"""
        completed_sessions = [s for s in self.sessions if s.end_time]
        if not completed_sessions:
            return 0
        
        total_duration = sum(
            (s.end_time - s.start_time).total_seconds()
            for s in completed_sessions
        )
        return total_duration / len(completed_sessions)
    
    def get_most_played_mode(self) -> Optional[str]:
        """Get the most played game mode"""
        if not self.sessions:
            return None
        
        mode_counts = {}
        for session in self.sessions:
            for mode, count in session.game_modes_played.items():
                mode_counts[mode] = mode_counts.get(mode, 0) + count
        
        return max(mode_counts.items(), key=lambda x: x[1])[0] if mode_counts else None
    
    def get_total_distance(self) -> float:
        """Calculate total distance driven across all sessions"""
        return sum(session.total_distance for session in self.sessions)
    
    def get_total_crashes(self) -> int:
        """Calculate total number of crashes across all sessions"""
        return sum(session.total_crashes for session in self.sessions)
    
    def get_economy_stats(self) -> Dict:
        """Get economy statistics"""
        total_earned = sum(session.total_coins_earned for session in self.sessions)
        total_spent = sum(session.total_coins_spent for session in self.sessions)
        return {
            'total_earned': total_earned,
            'total_spent': total_spent,
            'net_balance': total_earned - total_spent
        }
    
    def get_performance_stats(self) -> Dict:
        """Get average performance metrics"""
        if not self.sessions:
            return {}
        
        total_fps = 0
        total_frame_time = 0
        total_memory = 0
        count = 0
        
        for session in self.sessions:
            metrics = session.performance_metrics
            if metrics:
                total_fps += metrics.get('fps', 0)
                total_frame_time += metrics.get('frame_time', 0)
                total_memory += metrics.get('memory_usage', 0)
                count += 1
        
        if count == 0:
            return {}
        
        return {
            'average_fps': total_fps / count,
            'average_frame_time': total_frame_time / count,
            'average_memory_usage': total_memory / count
        }
    
    def get_error_stats(self) -> Dict[str, int]:
        """Get error statistics"""
        error_counts = {}
        for session in self.sessions:
            for event in session.events:
                if event.type == EventType.ERROR:
                    error_type = event.data.get('error_type', 'unknown')
                    error_counts[error_type] = error_counts.get(error_type, 0) + 1
        return error_counts 