import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import datetime
import json

class ProgressTracker:
    """Class to handle user progress tracking and analytics"""
    
    def __init__(self, user_id):
        """Initialize the progress tracker for a specific user"""
        self.user_id = user_id
        self.progress_data = self._load_progress_data()
    
    def _load_progress_data(self):
        """Load progress data from session state or initialize if not exists"""
        if 'user_progress' not in st.session_state:
            st.session_state.user_progress = {}
        
        if self.user_id not in st.session_state.user_progress:
            st.session_state.user_progress[self.user_id] = {
                'completed_topics': [],
                'problem_attempts': [],
                'achievements': [],
                'learning_streak': 0,
                'last_active': None
            }
        
        return st.session_state.user_progress[self.user_id]
    
    def save_progress(self):
        """Save the current progress to session state"""
        st.session_state.user_progress[self.user_id] = self.progress_data
    
    def mark_topic_completed(self, topic_id, category):
        """Mark a topic as completed"""
        # Check if already completed
        for completed in self.progress_data['completed_topics']:
            if completed['topic_id'] == topic_id and completed['category'] == category:
                # Update timestamp if already exists
                completed['timestamp'] = datetime.datetime.now().isoformat()
                self.save_progress()
                return
        
        # Add new completed topic
        self.progress_data['completed_topics'].append({
            'topic_id': topic_id,
            'category': category,
            'timestamp': datetime.datetime.now().isoformat()
        })
        
        # Update streak
        self._update_streak()
        
        # Check for achievements
        self._check_topic_achievements(category)
        
        self.save_progress()
    
    def record_problem_attempt(self, problem_id, topic_id, category, success, code=None):
        """Record a problem attempt"""
        self.progress_data['problem_attempts'].append({
            'problem_id': problem_id,
            'topic_id': topic_id,
            'category': category,
            'success': success,
            'timestamp': datetime.datetime.now().isoformat(),
            'code': code
        })
        
        # Update streak
        self._update_streak()
        
        # Check for achievements
        self._check_problem_achievements()
        
        self.save_progress()
    
    def _update_streak(self):
        """Update the learning streak"""
        today = datetime.datetime.now().date()
        
        if self.progress_data['last_active'] is None:
            # First activity
            self.progress_data['learning_streak'] = 1
        else:
            last_active = datetime.datetime.fromisoformat(self.progress_data['last_active']).date()
            
            # If last active was yesterday, increment streak
            if (today - last_active).days == 1:
                self.progress_data['learning_streak'] += 1
            # If last active was today, keep streak
            elif (today - last_active).days == 0:
                pass
            # If more than a day has passed, reset streak
            else:
                self.progress_data['learning_streak'] = 1
        
        self.progress_data['last_active'] = datetime.datetime.now().isoformat()
    
    def _check_topic_achievements(self, category):
        """Check for topic-based achievements"""
        completed_topics = self.progress_data['completed_topics']
        
        # Count topics completed by category
        category_counts = {}
        for topic in completed_topics:
            cat = topic['category']
            if cat not in category_counts:
                category_counts[cat] = 0
            category_counts[cat] += 1
        
        # Define achievements to check
        achievements = [
            {'id': 'beginner_dsa', 'name': 'DSA Beginner', 'description': 'Complete 5 DSA topics', 
             'condition': lambda: category_counts.get('dsa', 0) >= 5, 'category': 'dsa'},
            {'id': 'intermediate_dsa', 'name': 'DSA Intermediate', 'description': 'Complete 10 DSA topics', 
             'condition': lambda: category_counts.get('dsa', 0) >= 10, 'category': 'dsa'},
            {'id': 'advanced_dsa', 'name': 'DSA Advanced', 'description': 'Complete 15 DSA topics', 
             'condition': lambda: category_counts.get('dsa', 0) >= 15, 'category': 'dsa'},
            
            {'id': 'beginner_sys', 'name': 'System Design Beginner', 'description': 'Complete 3 System Design topics', 
             'condition': lambda: category_counts.get('system_design', 0) >= 3, 'category': 'system_design'},
            {'id': 'intermediate_sys', 'name': 'System Design Intermediate', 'description': 'Complete 6 System Design topics', 
             'condition': lambda: category_counts.get('system_design', 0) >= 6, 'category': 'system_design'},
            
            {'id': 'beginner_math', 'name': 'Math Beginner', 'description': 'Complete 3 Math topics', 
             'condition': lambda: category_counts.get('math', 0) >= 3, 'category': 'math'},
            {'id': 'intermediate_math', 'name': 'Math Intermediate', 'description': 'Complete 6 Math topics', 
             'condition': lambda: category_counts.get('math', 0) >= 6, 'category': 'math'},
            
            {'id': 'all_rounder', 'name': 'All-Rounder', 'description': 'Complete at least 3 topics in each category', 
             'condition': lambda: all(category_counts.get(cat, 0) >= 3 for cat in ['dsa', 'system_design', 'math'])}
        ]
        
        # Check each achievement and award if conditions met
        for achievement in achievements:
            if achievement['id'] not in [a['id'] for a in self.progress_data['achievements']]:
                if achievement['condition']():
                    self.award_achievement(achievement['id'], achievement['name'], achievement['description'])
    
    def _check_problem_achievements(self):
        """Check for problem-based achievements"""
        problem_attempts = self.progress_data['problem_attempts']
        
        # Count successful attempts
        successful_attempts = sum(1 for attempt in problem_attempts if attempt['success'])
        
        # Count successful attempts by category
        category_success = {}
        for attempt in problem_attempts:
            if attempt['success']:
                cat = attempt['category']
                if cat not in category_success:
                    category_success[cat] = 0
                category_success[cat] += 1
        
        # Define achievements to check
        achievements = [
            {'id': 'problem_solver_1', 'name': 'Problem Solver I', 'description': 'Successfully solve 5 problems', 
             'condition': lambda: successful_attempts >= 5},
            {'id': 'problem_solver_2', 'name': 'Problem Solver II', 'description': 'Successfully solve 25 problems', 
             'condition': lambda: successful_attempts >= 25},
            {'id': 'problem_solver_3', 'name': 'Problem Solver III', 'description': 'Successfully solve 50 problems', 
             'condition': lambda: successful_attempts >= 50},
            
            {'id': 'dsa_expert', 'name': 'DSA Expert', 'description': 'Successfully solve 20 DSA problems', 
             'condition': lambda: category_success.get('dsa', 0) >= 20, 'category': 'dsa'},
            {'id': 'system_design_expert', 'name': 'System Design Expert', 'description': 'Successfully solve 10 System Design problems', 
             'condition': lambda: category_success.get('system_design', 0) >= 10, 'category': 'system_design'},
            {'id': 'math_expert', 'name': 'Math Expert', 'description': 'Successfully solve 15 Math problems', 
             'condition': lambda: category_success.get('math', 0) >= 15, 'category': 'math'}
        ]
        
        # Check each achievement and award if conditions met
        for achievement in achievements:
            if achievement['id'] not in [a['id'] for a in self.progress_data['achievements']]:
                if achievement['condition']():
                    self.award_achievement(achievement['id'], achievement['name'], achievement['description'])
    
    def award_achievement(self, achievement_id, name, description):
        """Award an achievement to the user"""
        if achievement_id not in [a['id'] for a in self.progress_data['achievements']]:
            self.progress_data['achievements'].append({
                'id': achievement_id,
                'name': name,
                'description': description,
                'timestamp': datetime.datetime.now().isoformat()
            })
            self.save_progress()
            
            # Return True to indicate a new achievement was awarded
            return True
        
        return False
    
    def get_topic_completion_percentage(self, category=None):
        """Get the percentage of topics completed for a category"""
        # This would normally query the API for total topics, but we'll hardcode for demo
        total_topics = {
            'dsa': 20,
            'system_design': 12,
            'math': 24
        }
        
        completed_topics = self.progress_data['completed_topics']
        
        if category:
            completed_count = sum(1 for topic in completed_topics if topic['category'] == category)
            return (completed_count / total_topics[category]) * 100
        else:
            # Overall completion
            completed_count = len(completed_topics)
            total_count = sum(total_topics.values())
            return (completed_count / total_count) * 100
    
    def get_problem_success_rate(self, category=None):
        """Get the success rate for problem attempts"""
        problem_attempts = self.progress_data['problem_attempts']
        
        if category:
            category_attempts = [attempt for attempt in problem_attempts if attempt['category'] == category]
            if not category_attempts:
                return 0
            
            success_count = sum(1 for attempt in category_attempts if attempt['success'])
            return (success_count / len(category_attempts)) * 100
        else:
            if not problem_attempts:
                return 0
            
            success_count = sum(1 for attempt in problem_attempts if attempt['success'])
            return (success_count / len(problem_attempts)) * 100
    
    def get_activity_over_time(self):
        """Get activity data over time for visualization"""
        # Combine all activities
        activities = []
        
        # Add topic completions
        for topic in self.progress_data['completed_topics']:
            activities.append({
                'timestamp': datetime.datetime.fromisoformat(topic['timestamp']),
                'activity': 'Topic Completion',
                'category': topic['category'],
                'success': True
            })
        
        # Add problem attempts
        for attempt in self.progress_data['problem_attempts']:
            activities.append({
                'timestamp': datetime.datetime.fromisoformat(attempt['timestamp']),
                'activity': 'Problem Attempt',
                'category': attempt['category'],
                'success': attempt['success']
            })
        
        # Sort by timestamp
        activities.sort(key=lambda x: x['timestamp'])
        
        # Convert to dataframe for easier visualization
        if activities:
            df = pd.DataFrame(activities)
            return df
        else:
            return pd.DataFrame(columns=['timestamp', 'activity', 'category', 'success'])
    
    def render_progress_dashboard(self):
        """Render the progress dashboard"""
        st.markdown("## Your Learning Progress")
        
        # Display learning streak
        streak = self.progress_data['learning_streak']
        st.markdown(f"### 🔥 Learning Streak: {streak} {'day' if streak == 1 else 'days'}")
        
        # Show overall completion percentage
        overall_completion = self.get_topic_completion_percentage()
        
        # Create metrics for each category
        col1, col2, col3 = st.columns(3)
        
        with col1:
            dsa_completion = self.get_topic_completion_percentage('dsa')
            st.metric("DSA Topics", f"{dsa_completion:.1f}%")
            
            # Mini progress bar
            st.progress(dsa_completion / 100)
        
        with col2:
            sys_completion = self.get_topic_completion_percentage('system_design')
            st.metric("System Design Topics", f"{sys_completion:.1f}%")
            
            # Mini progress bar
            st.progress(sys_completion / 100)
        
        with col3:
            math_completion = self.get_topic_completion_percentage('math')
            st.metric("Math Topics", f"{math_completion:.1f}%")
            
            # Mini progress bar
            st.progress(math_completion / 100)
        
        # Problem success rates
        st.markdown("### Problem Success Rates")
        
        # Get problem success rates
        overall_success = self.get_problem_success_rate()
        dsa_success = self.get_problem_success_rate('dsa')
        sys_success = self.get_problem_success_rate('system_design')
        math_success = self.get_problem_success_rate('math')
        
        # Create a bar chart
        categories = ["Overall", "DSA", "System Design", "Math"]
        success_rates = [overall_success, dsa_success, sys_success, math_success]
        
        fig = go.Figure(data=[
            go.Bar(
                x=categories,
                y=success_rates,
                marker_color=['blue', 'green', 'orange', 'red']
            )
        ])
        
        fig.update_layout(
            title="Problem Success Rates by Category",
            xaxis_title="Category",
            yaxis_title="Success Rate (%)",
            yaxis=dict(range=[0, 100])
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Activity over time
        st.markdown("### Activity Over Time")
        
        # Get activity data
        activity_df = self.get_activity_over_time()
        
        if len(activity_df) > 0:
            # Group by day and count activities
            activity_df['date'] = activity_df['timestamp'].dt.date
            daily_counts = activity_df.groupby('date').size().reset_index(name='count')
            
            # Create line chart
            fig = px.line(
                daily_counts, 
                x='date', 
                y='count',
                title='Daily Learning Activities',
                labels={'date': 'Date', 'count': 'Number of Activities'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show success vs. failure for problem attempts
            problem_attempts = activity_df[activity_df['activity'] == 'Problem Attempt']
            
            if len(problem_attempts) > 0:
                success_count = sum(problem_attempts['success'])
                failure_count = len(problem_attempts) - success_count
                
                # Create pie chart
                fig = go.Figure(data=[go.Pie(
                    labels=['Success', 'Failure'],
                    values=[success_count, failure_count],
                    marker_colors=['green', 'red']
                )])
                
                fig.update_layout(title="Problem Attempt Outcomes")
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No activity data available yet. Start learning to see your progress!")
        
        # Achievements section
        st.markdown("### 🏆 Achievements")
        
        achievements = self.progress_data['achievements']
        
        if achievements:
            # Display in a grid
            achievements_per_row = 3
            rows = (len(achievements) + achievements_per_row - 1) // achievements_per_row
            
            for row in range(rows):
                cols = st.columns(achievements_per_row)
                for i in range(achievements_per_row):
                    idx = row * achievements_per_row + i
                    if idx < len(achievements):
                        achievement = achievements[idx]
                        with cols[i]:
                            st.markdown(f"""
                            <div style="padding: 1rem; border-radius: 0.5rem; background-color: #f0f8ff; border-left: 4px solid #1e90ff;">
                                <h4 style="color: #1e90ff;">🏆 {achievement['name']}</h4>
                                <p>{achievement['description']}</p>
                                <small>Earned on: {datetime.datetime.fromisoformat(achievement['timestamp']).strftime('%Y-%m-%d')}</small>
                            </div>
                            """, unsafe_allow_html=True)
        else:
            st.info("No achievements yet. Keep learning to earn badges!")
        
        # Recommended next steps
        st.markdown("### 📋 Recommended Next Steps")
        
        # Get completed topic IDs
        completed_ids = [topic['topic_id'] for topic in self.progress_data['completed_topics']]
        
        # Logic for recommendations (simplified)
        if 'arrays' in completed_ids and 'linked_lists' not in completed_ids:
            st.markdown("- **Linked Lists**: A natural progression after arrays")
        
        if 'stacks' in completed_ids and 'queues' not in completed_ids:
            st.markdown("- **Queues**: Complete your understanding of basic data structures")
        
        if 'calculus_limits' in completed_ids and 'calculus_derivatives' not in completed_ids:
            st.markdown("- **Derivatives**: Continue your calculus journey")
        
        if 'system_design_fundamentals' in completed_ids and 'scalability' not in completed_ids:
            st.markdown("- **Scalability**: Learn how to scale your systems")
        
        if dsa_completion < 30:
            st.markdown("- **Focus on DSA fundamentals**: Build a strong foundation")
        
        if overall_completion > 50 and sys_completion < 40:
            st.markdown("- **Explore more System Design topics**: Balance your knowledge")
    
    def render_learning_path(self):
        """Render a personalized learning path based on user progress"""
        st.markdown("## Your Personalized Learning Path")
        
        # Get completed topic IDs
        completed_ids = [topic['topic_id'] for topic in self.progress_data['completed_topics']]
        
        # Get problem attempts
        problem_attempts = self.progress_data['problem_attempts']
        problem_success = {attempt['topic_id']: attempt['success'] 
                          for attempt in problem_attempts if attempt['success']}
        
        # Define learning paths (simplified)
        dsa_path = [
            {'id': 'arrays', 'name': 'Arrays', 'level': 'beginner'},
            {'id': 'linked_lists', 'name': 'Linked Lists', 'level': 'beginner'},
            {'id': 'stacks', 'name': 'Stacks', 'level': 'beginner'},
            {'id': 'queues', 'name': 'Queues', 'level': 'beginner'},
            {'id': 'hash_tables', 'name': 'Hash Tables', 'level': 'intermediate'},
            {'id': 'trees', 'name': 'Trees', 'level': 'intermediate'},
            {'id': 'graphs', 'name': 'Graphs', 'level': 'advanced'},
            {'id': 'sorting', 'name': 'Sorting Algorithms', 'level': 'intermediate'},
            {'id': 'searching', 'name': 'Searching Algorithms', 'level': 'intermediate'},
            {'id': 'dynamic_programming', 'name': 'Dynamic Programming', 'level': 'advanced'}
        ]
        
        system_design_path = [
            {'id': 'system_design_fundamentals', 'name': 'System Design Fundamentals', 'level': 'beginner'},
            {'id': 'scalability', 'name': 'Scalability', 'level': 'beginner'},
            {'id': 'load_balancing', 'name': 'Load Balancing', 'level': 'beginner'},
            {'id': 'caching', 'name': 'Caching', 'level': 'intermediate'},
            {'id': 'database_design', 'name': 'Database Design', 'level': 'intermediate'},
            {'id': 'microservices', 'name': 'Microservices', 'level': 'intermediate'},
            {'id': 'distributed_systems', 'name': 'Distributed Systems', 'level': 'advanced'}
        ]
        
        math_path = [
            {'id': 'calculus_limits', 'name': 'Limits', 'level': 'beginner'},
            {'id': 'calculus_derivatives', 'name': 'Derivatives', 'level': 'beginner'},
            {'id': 'calculus_integrals', 'name': 'Integrals', 'level': 'intermediate'},
            {'id': 'linear_algebra_vectors', 'name': 'Vectors', 'level': 'beginner'},
            {'id': 'linear_algebra_matrices', 'name': 'Matrices', 'level': 'intermediate'},
            {'id': 'statistics_descriptive', 'name': 'Descriptive Statistics', 'level': 'beginner'},
            {'id': 'statistics_probability', 'name': 'Probability', 'level': 'intermediate'}
        ]
        
        # Mark items as completed
        for path in [dsa_path, system_design_path, math_path]:
            for item in path:
                item['completed'] = item['id'] in completed_ids
                item['problem_solved'] = item['id'] in problem_success
        
        # Calculate next recommended items
        def get_next_items(path, count=3):
            # First prioritize incomplete items based on their order
            incomplete = [item for item in path if not item['completed']]
            return incomplete[:count]
        
        # Get next items for each path
        next_dsa = get_next_items(dsa_path)
        next_system = get_next_items(system_design_path)
        next_math = get_next_items(math_path)
        
        # Create tabs for different paths
        tab1, tab2, tab3 = st.tabs(["DSA Path", "System Design Path", "Math Path"])
        
        with tab1:
            st.markdown("### Data Structures & Algorithms Path")
            self._render_path_items(dsa_path, "dsa")
        
        with tab2:
            st.markdown("### System Design Path")
            self._render_path_items(system_design_path, "system_design")
        
        with tab3:
            st.markdown("### Mathematics Path")
            self._render_path_items(math_path, "math")
        
        # Overall recommendations
        st.markdown("### 🌟 Recommended Next Steps")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if next_dsa:
                item = next_dsa[0]
                st.markdown(f"""
                <div style="padding: 1rem; border-radius: 0.5rem; background-color: #f8f9fa; border-left: 4px solid #28a745;">
                    <h4>Next DSA Topic</h4>
                    <p><strong>{item['name']}</strong> ({item['level']})</p>
                    <button style="background-color: #28a745; color: white; border: none; padding: 0.5rem 1rem; border-radius: 0.25rem;">
                        Start Learning
                    </button>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            if next_system:
                item = next_system[0]
                st.markdown(f"""
                <div style="padding: 1rem; border-radius: 0.5rem; background-color: #f8f9fa; border-left: 4px solid #fd7e14;">
                    <h4>Next System Design Topic</h4>
                    <p><strong>{item['name']}</strong> ({item['level']})</p>
                    <button style="background-color: #fd7e14; color: white; border: none; padding: 0.5rem 1rem; border-radius: 0.25rem;">
                        Start Learning
                    </button>
                </div>
                """, unsafe_allow_html=True)
        
        with col3:
            if next_math:
                item = next_math[0]
                st.markdown(f"""
                <div style="padding: 1rem; border-radius: 0.5rem; background-color: #f8f9fa; border-left: 4px solid #007bff;">
                    <h4>Next Math Topic</h4>
                    <p><strong>{item['name']}</strong> ({item['level']})</p>
                    <button style="background-color: #007bff; color: white; border: none; padding: 0.5rem 1rem; border-radius: 0.25rem;">
                        Start Learning
                    </button>
                </div>
                """, unsafe_allow_html=True)
    
    def _render_path_items(self, path_items, category):
        """Render items in a learning path"""
        # Display as a timeline
        for i, item in enumerate(path_items):
            # Determine status and styling
            if item['completed']:
                status = "✅ Completed"
                color = "#28a745"
                bg_color = "#f0fff0"
            else:
                status = "⏳ Not Started"
                color = "#6c757d"
                bg_color = "#f8f9fa"
            
            # Add extra indicator if problem was solved
            problem_indicator = "🏆 Problem Solved" if item.get('problem_solved') else ""
            
            # Create timeline item
            st.markdown(f"""
            <div style="display: flex; margin-bottom: 1rem;">
                <div style="margin-right: 1rem; display: flex; flex-direction: column; align-items: center;">
                    <div style="width: 30px; height: 30px; border-radius: 50%; background-color: {color}; color: white; display: flex; justify-content: center; align-items: center;">
                        {i+1}
                    </div>
                    {'' if i == len(path_items)-1 else '<div style="width: 2px; flex-grow: 1; background-color: #6c757d; margin-top: 0.5rem;"></div>'}
                </div>
                <div style="flex-grow: 1; padding: 1rem; border-radius: 0.5rem; background-color: {bg_color}; border-left: 4px solid {color};">
                    <h4 style="margin-top: 0;">{item['name']} <span style="font-size: 0.8rem; color: #6c757d;">({item['level']})</span></h4>
                    <p style="margin-bottom: 0.5rem;">{status} {problem_indicator}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)