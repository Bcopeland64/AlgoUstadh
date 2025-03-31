import streamlit as st
import datetime
import pandas as pd
import time

class CommunityForums:
    """Class to handle community forums and discussions"""
    
    def __init__(self, user_id):
        """Initialize the community forums for a user"""
        self.user_id = user_id
        self._initialize_forums_data()
    
    def _initialize_forums_data(self):
        """Initialize forum data in session state if not exists"""
        if 'forum_data' not in st.session_state:
            # Create sample forum data
            sample_posts = [
                {
                    'id': 1,
                    'title': 'Help with recursive backtracking algorithm',
                    'content': 'I\'m having trouble understanding how to implement backtracking for the N-Queens problem. Can someone explain the recursive approach?',
                    'author': 'alex_dev',
                    'timestamp': datetime.datetime.now() - datetime.timedelta(days=3),
                    'category': 'dsa',
                    'topic': 'backtracking',
                    'likes': 5,
                    'views': 42,
                    'comments': [
                        {
                            'id': 1,
                            'content': 'The key insight for backtracking in N-Queens is to place queens one by one in different rows. When placing a queen, check if it\'s safe from attack by previously placed queens, then recursively place the rest.',
                            'author': 'code_master',
                            'timestamp': datetime.datetime.now() - datetime.timedelta(days=2, hours=12),
                            'likes': 3
                        },
                        {
                            'id': 2,
                            'content': 'Here\'s a good tutorial that helped me: [link to resource]',
                            'author': 'learning_daily',
                            'timestamp': datetime.datetime.now() - datetime.timedelta(days=2),
                            'likes': 2
                        }
                    ]
                },
                {
                    'id': 2,
                    'title': 'Database sharding strategies for high-volume applications',
                    'content': 'I\'m working on a system that needs to handle millions of transactions per day. What are the best practices for database sharding in this scenario?',
                    'author': 'system_architect',
                    'timestamp': datetime.datetime.now() - datetime.timedelta(days=5),
                    'category': 'system_design',
                    'topic': 'database_design',
                    'likes': 12,
                    'views': 87,
                    'comments': [
                        {
                            'id': 3,
                            'content': 'For high-volume transactional systems, consider hash-based sharding with consistent hashing to distribute load evenly and minimize resharding operations when adding new nodes.',
                            'author': 'db_expert',
                            'timestamp': datetime.datetime.now() - datetime.timedelta(days=4),
                            'likes': 8
                        },
                        {
                            'id': 4,
                            'content': 'Don\'t forget about cross-shard transactions - they can be tricky. You might want to consider a design that minimizes them.',
                            'author': 'distributed_systems_dev',
                            'timestamp': datetime.datetime.now() - datetime.timedelta(days=3),
                            'likes': 5
                        },
                        {
                            'id': 5,
                            'content': 'We used geographic sharding for our application and it worked well for localized access patterns.',
                            'author': 'cloud_architect',
                            'timestamp': datetime.datetime.now() - datetime.timedelta(days=2),
                            'likes': 3
                        }
                    ]
                },
                {
                    'id': 3,
                    'title': 'Understanding eigenvalues in machine learning applications',
                    'content': 'I\'m struggling to understand the practical significance of eigenvalues in PCA and other ML algorithms. Can someone explain in simple terms?',
                    'author': 'ml_beginner',
                    'timestamp': datetime.datetime.now() - datetime.timedelta(days=1),
                    'category': 'math',
                    'topic': 'linear_algebra_eigen',
                    'likes': 7,
                    'views': 29,
                    'comments': [
                        {
                            'id': 6,
                            'content': 'Eigenvalues in PCA tell you how much variance is captured by each principal component. Larger eigenvalues correspond to directions with greater variance in your data.',
                            'author': 'stats_phd',
                            'timestamp': datetime.datetime.now() - datetime.timedelta(hours=20),
                            'likes': 4
                        }
                    ]
                },
                {
                    'id': 4,
                    'title': 'Tips for solving dynamic programming problems',
                    'content': 'I always struggle with recognizing and solving DP problems in coding interviews. Any tips or frameworks that can help?',
                    'author': 'interview_prep',
                    'timestamp': datetime.datetime.now() - datetime.timedelta(days=7),
                    'category': 'dsa',
                    'topic': 'dynamic_programming',
                    'likes': 21,
                    'views': 136,
                    'comments': [
                        {
                            'id': 7,
                            'content': 'I find it helpful to look for overlapping subproblems and optimal substructure. Ask yourself: "Can I build the solution from solutions to smaller instances of the same problem?"',
                            'author': 'algo_enthusiast',
                            'timestamp': datetime.datetime.now() - datetime.timedelta(days=6),
                            'likes': 15
                        },
                        {
                            'id': 8,
                            'content': 'Practice these patterns: 0/1 Knapsack, Unbounded Knapsack, Shortest Path, Longest Common Subsequence, and Edit Distance. Many DP problems are variations of these.',
                            'author': 'faang_dev',
                            'timestamp': datetime.datetime.now() - datetime.timedelta(days=5),
                            'likes': 12
                        },
                        {
                            'id': 9,
                            'content': 'Start by solving the problem using recursion + memoization, then convert to a bottom-up approach if needed.',
                            'author': 'code_teacher',
                            'timestamp': datetime.datetime.now() - datetime.timedelta(days=4),
                            'likes': 8
                        }
                    ]
                },
                {
                    'id': 5,
                    'title': 'Load balancing strategies for microservices',
                    'content': 'What are the pros and cons of different load balancing algorithms when dealing with a large microservices architecture?',
                    'author': 'cloud_native_dev',
                    'timestamp': datetime.datetime.now() - datetime.timedelta(days=2),
                    'category': 'system_design',
                    'topic': 'load_balancing',
                    'likes': 9,
                    'views': 52,
                    'comments': [
                        {
                            'id': 10,
                            'content': 'Round robin is simple but doesn\'t account for server load. Least connections works better when some requests take longer than others. Weighted algorithms are good when your instances have different capacities.',
                            'author': 'devops_guru',
                            'timestamp': datetime.datetime.now() - datetime.timedelta(days=1),
                            'likes': 6
                        },
                        {
                            'id': 11,
                            'content': 'Don\'t forget about session affinity when stateful services are involved.',
                            'author': 'backend_dev',
                            'timestamp': datetime.datetime.now() - datetime.timedelta(hours=18),
                            'likes': 4
                        }
                    ]
                }
            ]
            
            st.session_state.forum_data = {
                'next_post_id': len(sample_posts) + 1,
                'next_comment_id': sum(len(post['comments']) for post in sample_posts) + 1,
                'posts': sample_posts
            }
        
        # Load the forum data
        self.forum_data = st.session_state.forum_data
    
    def save_forum_data(self):
        """Save forum data to session state"""
        st.session_state.forum_data = self.forum_data
    
    def get_posts(self, category=None, topic=None, search_term=None):
        """Get forum posts, optionally filtered by category, topic, or search term"""
        posts = self.forum_data['posts']
        
        # Apply filters
        if category:
            posts = [post for post in posts if post['category'] == category]
        
        if topic:
            posts = [post for post in posts if post['topic'] == topic]
        
        if search_term:
            search_term = search_term.lower()
            posts = [post for post in posts if 
                    search_term in post['title'].lower() or 
                    search_term in post['content'].lower()]
        
        # Sort by timestamp (newest first)
        posts = sorted(posts, key=lambda x: x['timestamp'], reverse=True)
        
        return posts
    
    def get_post(self, post_id):
        """Get a specific post by ID"""
        for post in self.forum_data['posts']:
            if post['id'] == post_id:
                return post
        return None
    
    def create_post(self, title, content, category, topic):
        """Create a new forum post"""
        # Get next post ID
        post_id = self.forum_data['next_post_id']
        self.forum_data['next_post_id'] += 1
        
        # Create the post
        new_post = {
            'id': post_id,
            'title': title,
            'content': content,
            'author': self.user_id,
            'timestamp': datetime.datetime.now(),
            'category': category,
            'topic': topic,
            'likes': 0,
            'views': 0,
            'comments': []
        }
        
        # Add to posts
        self.forum_data['posts'].append(new_post)
        
        # Save changes
        self.save_forum_data()
        
        return post_id
    
    def add_comment(self, post_id, content):
        """Add a comment to a post"""
        post = self.get_post(post_id)
        if not post:
            return False
        
        # Get next comment ID
        comment_id = self.forum_data['next_comment_id']
        self.forum_data['next_comment_id'] += 1
        
        # Create the comment
        new_comment = {
            'id': comment_id,
            'content': content,
            'author': self.user_id,
            'timestamp': datetime.datetime.now(),
            'likes': 0
        }
        
        # Add to post's comments
        post['comments'].append(new_comment)
        
        # Save changes
        self.save_forum_data()
        
        return True
    
    def like_post(self, post_id):
        """Like a post"""
        post = self.get_post(post_id)
        if post:
            post['likes'] += 1
            self.save_forum_data()
            return True
        return False
    
    def view_post(self, post_id):
        """Increment view count for a post"""
        post = self.get_post(post_id)
        if post:
            post['views'] += 1
            self.save_forum_data()
            return True
        return False
    
    def like_comment(self, post_id, comment_id):
        """Like a comment"""
        post = self.get_post(post_id)
        if not post:
            return False
        
        for comment in post['comments']:
            if comment['id'] == comment_id:
                comment['likes'] += 1
                self.save_forum_data()
                return True
        
        return False
    
    def render_forum_list(self, category=None):
        """Render the forum post list, optionally filtered by category"""
        st.markdown("## Community Forums")
        
        # Forum filters
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            selected_category = st.selectbox(
                "Category", 
                ["All Categories", "Data Structures & Algorithms", "System Design", "Mathematics"],
                index=0 if not category else ["All Categories", "Data Structures & Algorithms", "System Design", "Mathematics"].index(category)
            )
            
            # Convert display names to internal category IDs
            category_map = {
                "All Categories": None,
                "Data Structures & Algorithms": "dsa",
                "System Design": "system_design",
                "Mathematics": "math"
            }
            filter_category = category_map[selected_category]
        
        with col2:
            search_term = st.text_input("Search posts", "")
        
        with col3:
            st.write("")
            st.write("")
            if st.button("New Post", type="primary"):
                st.session_state.forum_view = "new_post"
                st.session_state.forum_category = filter_category
                st.rerun()
        
        # Get filtered posts
        posts = self.get_posts(category=filter_category, search_term=search_term)
        
        if not posts:
            st.info("No posts found. Be the first to start a discussion!")
        else:
            # Display posts
            for post in posts:
                # Format the post as a card
                with st.container():
                    col1, col2 = st.columns([5, 1])
                    
                    with col1:
                        # Title with link
                        if st.button(post['title'], key=f"post_{post['id']}", type="text"):
                            # View the post
                            self.view_post(post['id'])
                            st.session_state.forum_view = "post"
                            st.session_state.forum_post_id = post['id']
                            st.rerun()
                        
                        # Post metadata
                        category_display = {
                            "dsa": "Data Structures & Algorithms",
                            "system_design": "System Design",
                            "math": "Mathematics"
                        }
                        st.markdown(f"By **{post['author']}** in *{category_display.get(post['category'], post['category'])}* • {post['timestamp'].strftime('%b %d, %Y')}")
                    
                    with col2:
                        # Stats
                        st.markdown(f"👍 {post['likes']} • 💬 {len(post['comments'])} • 👁️ {post['views']}")
                    
                    st.markdown("---")
    
    def render_post_view(self, post_id):
        """Render a single post view with comments"""
        post = self.get_post(post_id)
        if not post:
            st.error("Post not found")
            return
        
        # Back button
        if st.button("← Back to Forums"):
            st.session_state.forum_view = "list"
            st.rerun()
        
        # Post header
        st.markdown(f"# {post['title']}")
        
        # Post metadata
        category_display = {
            "dsa": "Data Structures & Algorithms",
            "system_design": "System Design",
            "math": "Mathematics"
        }
        st.markdown(f"By **{post['author']}** in *{category_display.get(post['category'], post['category'])}* • {post['timestamp'].strftime('%b %d, %Y at %I:%M %p')}")
        
        # Post content
        st.markdown(f"""
        <div style="padding: 1rem; border-radius: 0.5rem; background-color: #f8f9fa; margin: 1rem 0;">
            {post['content']}
        </div>
        """, unsafe_allow_html=True)
        
        # Post actions
        col1, col2, col3 = st.columns([1, 1, 3])
        
        with col1:
            if st.button(f"👍 Like ({post['likes']})", key="like_post"):
                self.like_post(post_id)
                st.rerun()
        
        with col2:
            st.markdown(f"👁️ {post['views']} views")
        
        st.markdown("---")
        
        # Comments section
        st.markdown(f"## Comments ({len(post['comments'])})")
        
        if not post['comments']:
            st.info("No comments yet. Be the first to comment!")
        
        # Display comments
        for comment in sorted(post['comments'], key=lambda x: x['timestamp']):
            with st.container():
                st.markdown(f"""
                <div style="padding: 0.8rem; border-left: 3px solid #3b82f6; background-color: #f0f7ff; margin-bottom: 1rem; border-radius: 0.3rem;">
                    <p style="margin-bottom: 0.5rem;">{comment['content']}</p>
                    <div style="display: flex; justify-content: space-between; font-size: 0.8rem; color: #6b7280;">
                        <span>By <strong>{comment['author']}</strong> • {comment['timestamp'].strftime('%b %d, %Y at %I:%M %p')}</span>
                        <span>👍 {comment['likes']}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Like comment button
                if st.button(f"Like", key=f"like_comment_{comment['id']}"):
                    self.like_comment(post_id, comment['id'])
                    st.rerun()
        
        # Add comment form
        st.markdown("### Add a Comment")
        comment_text = st.text_area("Comment", "", key="comment_text")
        
        if st.button("Submit Comment"):
            if comment_text.strip():
                self.add_comment(post_id, comment_text)
                st.success("Comment added!")
                time.sleep(1)  # Give time for the success message to be seen
                st.rerun()
            else:
                st.warning("Please enter a comment")
    
    def render_new_post_form(self):
        """Render the form to create a new post"""
        st.markdown("## Create New Forum Post")
        
        # Back button
        if st.button("← Back to Forums"):
            st.session_state.forum_view = "list"
            st.rerun()
        
        # Post form
        post_title = st.text_input("Title", "")
        
        category_options = {
            "dsa": "Data Structures & Algorithms",
            "system_design": "System Design",
            "math": "Mathematics"
        }
        
        post_category = st.selectbox(
            "Category",
            list(category_options.keys()),
            format_func=lambda x: category_options[x],
            index=0 if not hasattr(st.session_state, 'forum_category') or not st.session_state.forum_category else list(category_options.keys()).index(st.session_state.forum_category)
        )
        
        # Topic selection based on category
        topic_options = {
            "dsa": [
                "arrays", "linked_lists", "stacks", "queues", "hash_tables", "trees", 
                "binary_search_trees", "heaps", "graphs", "sorting", "searching", 
                "dynamic_programming", "backtracking", "greedy_algorithms"
            ],
            "system_design": [
                "system_design_fundamentals", "scalability", "load_balancing", "caching",
                "database_design", "microservices", "api_design", "message_queues",
                "distributed_systems", "fault_tolerance", "cap_theorem"
            ],
            "math": [
                "calculus_limits", "calculus_derivatives", "calculus_integrals",
                "linear_algebra_vectors", "linear_algebra_matrices", "linear_algebra_eigen",
                "statistics_descriptive", "statistics_probability", "statistics_inference",
                "discrete_math_logic", "discrete_math_sets", "discrete_math_combinatorics"
            ]
        }
        
        # Format function for readable topics
        def format_topic(topic):
            return topic.replace('_', ' ').title()
        
        post_topic = st.selectbox(
            "Topic",
            topic_options.get(post_category, []),
            format_func=format_topic
        )
        
        post_content = st.text_area("Content", "", height=200)
        
        if st.button("Create Post", type="primary"):
            if post_title.strip() and post_content.strip():
                post_id = self.create_post(post_title, post_content, post_category, post_topic)
                st.success("Post created successfully!")
                time.sleep(1)  # Give time for the success message to be seen
                st.session_state.forum_view = "post"
                st.session_state.forum_post_id = post_id
                st.rerun()
            else:
                st.warning("Please fill in all fields")
    
    def render_forums(self):
        """Main entry point to render the forum UI"""
        # Initialize view state if not exists
        if 'forum_view' not in st.session_state:
            st.session_state.forum_view = "list"
        
        # Render the appropriate view
        if st.session_state.forum_view == "list":
            self.render_forum_list()
        elif st.session_state.forum_view == "post" and hasattr(st.session_state, 'forum_post_id'):
            self.render_post_view(st.session_state.forum_post_id)
        elif st.session_state.forum_view == "new_post":
            self.render_new_post_form()
        else:
            # Default to list view
            self.render_forum_list()