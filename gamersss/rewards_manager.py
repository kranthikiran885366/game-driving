import json
import time
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta

class AchievementType(Enum):
    DISTANCE = "distance"
    SPEED = "speed"
    SKILL = "skill"
    COLLECTION = "collection"
    SOCIAL = "social"

@dataclass
class Achievement:
    id: str
    name: str
    description: str
    type: AchievementType
    requirement: int
    reward: int
    unlocked: bool = False
    progress: int = 0

@dataclass
class DailyReward:
    day: int
    coins: int
    item: Optional[str] = None
    claimed: bool = False

@dataclass
class Challenge:
    id: str
    name: str
    description: str
    requirement: int
    reward: int
    deadline: datetime
    completed: bool = False
    progress: int = 0

class RewardsManager:
    def __init__(self):
        self.coins = 0
        self.achievements: Dict[str, Achievement] = {}
        self.daily_rewards: List[DailyReward] = []
        self.challenges: List[Challenge] = []
        self.last_daily_claim: Optional[datetime] = None
        self._load_data()
        self._initialize_achievements()
        self._initialize_daily_rewards()
    
    def _load_data(self):
        try:
            with open('rewards_data.json', 'r') as f:
                data = json.load(f)
                self.coins = data.get('coins', 0)
                self.last_daily_claim = datetime.fromisoformat(data.get('last_daily_claim', '')) if data.get('last_daily_claim') else None
                
                # Load achievements
                for ach_data in data.get('achievements', []):
                    achievement = Achievement(
                        id=ach_data['id'],
                        name=ach_data['name'],
                        description=ach_data['description'],
                        type=AchievementType(ach_data['type']),
                        requirement=ach_data['requirement'],
                        reward=ach_data['reward'],
                        unlocked=ach_data.get('unlocked', False),
                        progress=ach_data.get('progress', 0)
                    )
                    self.achievements[achievement.id] = achievement
                
                # Load daily rewards
                for reward_data in data.get('daily_rewards', []):
                    reward = DailyReward(
                        day=reward_data['day'],
                        coins=reward_data['coins'],
                        item=reward_data.get('item'),
                        claimed=reward_data.get('claimed', False)
                    )
                    self.daily_rewards.append(reward)
                
                # Load challenges
                for challenge_data in data.get('challenges', []):
                    challenge = Challenge(
                        id=challenge_data['id'],
                        name=challenge_data['name'],
                        description=challenge_data['description'],
                        requirement=challenge_data['requirement'],
                        reward=challenge_data['reward'],
                        deadline=datetime.fromisoformat(challenge_data['deadline']),
                        completed=challenge_data.get('completed', False),
                        progress=challenge_data.get('progress', 0)
                    )
                    self.challenges.append(challenge)
        except FileNotFoundError:
            self._save_data()
    
    def _save_data(self):
        data = {
            'coins': self.coins,
            'last_daily_claim': self.last_daily_claim.isoformat() if self.last_daily_claim else None,
            'achievements': [
                {
                    'id': ach.id,
                    'name': ach.name,
                    'description': ach.description,
                    'type': ach.type.value,
                    'requirement': ach.requirement,
                    'reward': ach.reward,
                    'unlocked': ach.unlocked,
                    'progress': ach.progress
                }
                for ach in self.achievements.values()
            ],
            'daily_rewards': [
                {
                    'day': reward.day,
                    'coins': reward.coins,
                    'item': reward.item,
                    'claimed': reward.claimed
                }
                for reward in self.daily_rewards
            ],
            'challenges': [
                {
                    'id': challenge.id,
                    'name': challenge.name,
                    'description': challenge.description,
                    'requirement': challenge.requirement,
                    'reward': challenge.reward,
                    'deadline': challenge.deadline.isoformat(),
                    'completed': challenge.completed,
                    'progress': challenge.progress
                }
                for challenge in self.challenges
            ]
        }
        
        with open('rewards_data.json', 'w') as f:
            json.dump(data, f, indent=4)
    
    def _initialize_achievements(self):
        if not self.achievements:
            achievements = [
                Achievement(
                    id="distance_100",
                    name="Road Warrior",
                    description="Drive 100km",
                    type=AchievementType.DISTANCE,
                    requirement=100,
                    reward=1000
                ),
                Achievement(
                    id="speed_200",
                    name="Speed Demon",
                    description="Reach 200 km/h",
                    type=AchievementType.SPEED,
                    requirement=200,
                    reward=2000
                ),
                Achievement(
                    id="perfect_park",
                    name="Parking Master",
                    description="Complete 10 perfect parks",
                    type=AchievementType.SKILL,
                    requirement=10,
                    reward=1500
                ),
                Achievement(
                    id="collect_cars",
                    name="Car Collector",
                    description="Unlock 5 different vehicles",
                    type=AchievementType.COLLECTION,
                    requirement=5,
                    reward=3000
                ),
                Achievement(
                    id="social_butterfly",
                    name="Social Butterfly",
                    description="Complete 5 multiplayer races",
                    type=AchievementType.SOCIAL,
                    requirement=5,
                    reward=2000
                )
            ]
            
            for achievement in achievements:
                self.achievements[achievement.id] = achievement
    
    def _initialize_daily_rewards(self):
        if not self.daily_rewards:
            self.daily_rewards = [
                DailyReward(day=1, coins=100),
                DailyReward(day=2, coins=150),
                DailyReward(day=3, coins=200),
                DailyReward(day=4, coins=250),
                DailyReward(day=5, coins=300, item="nitro_boost"),
                DailyReward(day=6, coins=350),
                DailyReward(day=7, coins=500, item="special_skin")
            ]
    
    def add_coins(self, amount: int):
        """Add coins to the player's balance"""
        self.coins += amount
        self._save_data()
    
    def spend_coins(self, amount: int) -> bool:
        """Spend coins if enough are available"""
        if self.coins >= amount:
            self.coins -= amount
            self._save_data()
            return True
        return False
    
    def update_achievement_progress(self, achievement_id: str, progress: int):
        """Update progress for an achievement"""
        if achievement_id in self.achievements:
            achievement = self.achievements[achievement_id]
            if not achievement.unlocked:
                achievement.progress = min(progress, achievement.requirement)
                if achievement.progress >= achievement.requirement:
                    achievement.unlocked = True
                    self.add_coins(achievement.reward)
                self._save_data()
    
    def claim_daily_reward(self) -> Optional[Dict]:
        """Claim the next available daily reward"""
        if not self.last_daily_claim or (datetime.now() - self.last_daily_claim) >= timedelta(days=1):
            # Find the next unclaimed reward
            for reward in self.daily_rewards:
                if not reward.claimed:
                    reward.claimed = True
                    self.last_daily_claim = datetime.now()
                    self.add_coins(reward.coins)
                    self._save_data()
                    return {
                        'coins': reward.coins,
                        'item': reward.item
                    }
        return None
    
    def get_next_daily_reward(self) -> Optional[DailyReward]:
        """Get information about the next available daily reward"""
        if not self.last_daily_claim or (datetime.now() - self.last_daily_claim) >= timedelta(days=1):
            for reward in self.daily_rewards:
                if not reward.claimed:
                    return reward
        return None
    
    def update_challenge_progress(self, challenge_id: str, progress: int):
        """Update progress for a challenge"""
        for challenge in self.challenges:
            if challenge.id == challenge_id and not challenge.completed:
                challenge.progress = min(progress, challenge.requirement)
                if challenge.progress >= challenge.requirement:
                    challenge.completed = True
                    self.add_coins(challenge.reward)
                self._save_data()
                break
    
    def get_active_challenges(self) -> List[Challenge]:
        """Get list of active (uncompleted) challenges"""
        now = datetime.now()
        return [
            challenge for challenge in self.challenges
            if not challenge.completed and challenge.deadline > now
        ]
    
    def watch_ad_for_reward(self, reward_type: str) -> Optional[Dict]:
        """Simulate watching an ad for a reward"""
        rewards = {
            'double_coins': {'coins': 200, 'duration': 300},  # 5 minutes
            'revive': {'revives': 1},
            'refuel': {'fuel': 100}
        }
        
        if reward_type in rewards:
            return rewards[reward_type]
        return None
    
    def get_leaderboard_data(self, category: str) -> List[Dict]:
        """Get leaderboard data for a specific category"""
        # This would typically fetch data from a server
        # For now, return dummy data
        return [
            {'rank': 1, 'player': 'Player1', 'score': 1000},
            {'rank': 2, 'player': 'Player2', 'score': 900},
            {'rank': 3, 'player': 'Player3', 'score': 800}
        ]
    
    def get_achievement_progress(self) -> Dict[str, float]:
        """Get progress for all achievements as percentages"""
        return {
            ach_id: (ach.progress / ach.requirement) * 100
            for ach_id, ach in self.achievements.items()
        }
    
    def get_total_earned_coins(self) -> int:
        """Calculate total coins earned through achievements and challenges"""
        total = 0
        for achievement in self.achievements.values():
            if achievement.unlocked:
                total += achievement.reward
        for challenge in self.challenges:
            if challenge.completed:
                total += challenge.reward
        return total 