import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation
import networkx as nx
import io
import base64

# DSA Visualizations

def create_array_visualization(array_size=10, array_values=None):
    """Create an interactive array visualization"""
    if array_values is None:
        array_values = list(range(1, array_size + 1))
    
    # Create visualization
    fig = go.Figure()
    
    # Add array cells
    fig.add_trace(go.Scatter(
        x=list(range(len(array_values))),
        y=[1] * len(array_values),
        mode='markers+text',
        marker=dict(size=40, color='lightblue', line=dict(color='navy', width=2)),
        text=array_values,
        textposition="middle center",
        name='Array Values'
    ))
    
    # Add index markers below
    fig.add_trace(go.Scatter(
        x=list(range(len(array_values))),
        y=[0] * len(array_values),
        mode='text',
        text=list(range(len(array_values))),
        textposition="middle center",
        name='Indices',
        textfont=dict(color='darkblue')
    ))
    
    # Update layout
    fig.update_layout(
        title="Array Visualization",
        xaxis=dict(range=[-1, len(array_values)], showticklabels=False, zeroline=False),
        yaxis=dict(range=[-0.5, 1.5], showticklabels=False, zeroline=False),
        showlegend=False,
        height=200,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

def create_stack_visualization(stack_values=None):
    """Create an interactive stack visualization"""
    if stack_values is None:
        stack_values = [3, 7, 2, 9]
    
    # Create visualization
    fig = go.Figure()
    
    # Add stack elements
    for i, val in enumerate(stack_values):
        fig.add_trace(go.Scatter(
            x=[0, 1, 1, 0, 0],  # Box coordinates
            y=[i, i, i+0.8, i+0.8, i], 
            mode='lines',
            line=dict(color='navy', width=2),
            fill='toself',
            fillcolor='lightblue',
            hoveron='fills',
            name=f'Element {i}'
        ))
        
        # Add value text
        fig.add_trace(go.Scatter(
            x=[0.5],
            y=[i+0.4],
            mode='text',
            text=[str(val)],
            textposition="middle center",
            textfont=dict(size=14, color='black'),
            showlegend=False
        ))
    
    # Add "top" pointer
    if stack_values:
        fig.add_annotation(
            x=1.2,
            y=len(stack_values)-0.2,
            text="TOP",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor='red'
        )
    
    # Update layout
    fig.update_layout(
        title="Stack Visualization (LIFO)",
        xaxis=dict(range=[-0.5, 2], showticklabels=False, zeroline=False),
        yaxis=dict(range=[-0.5, len(stack_values) + 0.5], showticklabels=False, zeroline=False, autorange="reversed"),
        showlegend=False,
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

def create_queue_visualization(queue_values=None):
    """Create an interactive queue visualization"""
    if queue_values is None:
        queue_values = [5, 2, 8, 1]
    
    # Create visualization
    fig = go.Figure()
    
    # Add queue elements
    for i, val in enumerate(queue_values):
        fig.add_trace(go.Scatter(
            x=[i, i+1, i+1, i, i],  # Box coordinates
            y=[0, 0, 1, 1, 0], 
            mode='lines',
            line=dict(color='navy', width=2),
            fill='toself',
            fillcolor='lightblue',
            hoveron='fills',
            name=f'Element {i}'
        ))
        
        # Add value text
        fig.add_trace(go.Scatter(
            x=[i+0.5],
            y=[0.5],
            mode='text',
            text=[str(val)],
            textposition="middle center",
            textfont=dict(size=14, color='black'),
            showlegend=False
        ))
    
    # Add front and rear pointers
    if queue_values:
        fig.add_annotation(
            x=0.5,
            y=-0.3,
            text="FRONT",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor='red'
        )
        
        fig.add_annotation(
            x=len(queue_values)-0.5,
            y=-0.3,
            text="REAR",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor='green'
        )
    
    # Update layout
    fig.update_layout(
        title="Queue Visualization (FIFO)",
        xaxis=dict(range=[-0.5, len(queue_values) + 0.5], showticklabels=False, zeroline=False),
        yaxis=dict(range=[-1, 1.5], showticklabels=False, zeroline=False),
        showlegend=False,
        height=200,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

def create_hash_table_visualization(keys=None, values=None, table_size=10):
    """Create an interactive hash table visualization"""
    if keys is None:
        keys = ["apple", "banana", "cherry", "date", "fig"]
        values = [5, 8, 2, 10, 3]
    
    # Simple hash function for demo
    def simple_hash(key, size):
        if isinstance(key, str):
            return sum(ord(c) for c in key) % size
        return key % size
    
    # Create visualization
    fig = go.Figure()
    
    # Initialize empty buckets
    buckets = [[] for _ in range(table_size)]
    
    # Place items in buckets based on hash
    for i, (key, val) in enumerate(zip(keys, values)):
        index = simple_hash(key, table_size)
        buckets[index].append((key, val))
    
    # Draw the hash table
    for i in range(table_size):
        # Draw bucket
        fig.add_trace(go.Scatter(
            x=[0, 1, 1, 0, 0],
            y=[i, i, i+0.8, i+0.8, i],
            mode='lines',
            line=dict(color='navy', width=2),
            fill='toself',
            fillcolor='lightblue' if buckets[i] else 'white',
            name=f'Bucket {i}'
        ))
        
        # Add index
        fig.add_trace(go.Scatter(
            x=[-0.3],
            y=[i+0.4],
            mode='text',
            text=[f'{i}'],
            textposition="middle center",
            textfont=dict(size=14, color='black'),
            showlegend=False
        ))
        
        # Add content
        if buckets[i]:
            content_text = "<br>".join([f"{k}: {v}" for k, v in buckets[i]])
            fig.add_trace(go.Scatter(
                x=[0.5],
                y=[i+0.4],
                mode='text',
                text=[content_text],
                textposition="middle center",
                textfont=dict(size=12, color='black'),
                showlegend=False
            ))
    
    # Update layout
    fig.update_layout(
        title="Hash Table Visualization",
        xaxis=dict(range=[-0.5, 1.5], showticklabels=False, zeroline=False),
        yaxis=dict(range=[-0.5, table_size+0.5], showticklabels=False, zeroline=False, autorange="reversed"),
        showlegend=False,
        height=500,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

def create_binary_tree_visualization(tree_data=None):
    """Create an interactive binary tree visualization"""
    if tree_data is None:
        # Default binary tree for demo
        tree_data = {
            'value': 10,
            'left': {
                'value': 5,
                'left': {'value': 2, 'left': None, 'right': None},
                'right': {'value': 7, 'left': None, 'right': None}
            },
            'right': {
                'value': 15,
                'left': {'value': 12, 'left': None, 'right': None},
                'right': {'value': 18, 'left': None, 'right': None}
            }
        }
    
    # Create positions for tree nodes
    positions = {}
    labels = {}
    edges = []
    
    def traverse(node, x, y, level=0, parent=None):
        if not node:
            return
        
        # Adjust x position based on level
        spread = 2 ** (4 - level) / 2
        
        if 'id' not in node:
            node['id'] = len(positions)
        
        positions[node['id']] = (x, y)
        labels[node['id']] = str(node['value'])
        
        if parent is not None:
            edges.append((parent, node['id']))
        
        if node['left']:
            traverse(node['left'], x - spread, y - 1, level + 1, node['id'])
        if node['right']:
            traverse(node['right'], x + spread, y - 1, level + 1, node['id'])
    
    # Create the tree layout
    traverse(tree_data, 0, 0)
    
    # Create visualization
    fig = go.Figure()
    
    # Add edges (lines between nodes)
    for parent, child in edges:
        px, py = positions[parent]
        cx, cy = positions[child]
        fig.add_trace(go.Scatter(
            x=[px, cx],
            y=[py, cy],
            mode='lines',
            line=dict(color='black', width=1),
            showlegend=False
        ))
    
    # Add nodes
    node_x = [pos[0] for pos in positions.values()]
    node_y = [pos[1] for pos in positions.values()]
    node_text = [labels[node_id] for node_id in positions.keys()]
    
    fig.add_trace(go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        marker=dict(size=30, color='lightblue', line=dict(color='navy', width=2)),
        text=node_text,
        textposition="middle center",
        name='Nodes'
    ))
    
    # Update layout
    fig.update_layout(
        title="Binary Tree Visualization",
        xaxis=dict(range=[min(node_x)-1, max(node_x)+1], showticklabels=False, zeroline=False),
        yaxis=dict(range=[min(node_y)-1, max(node_y)+1], showticklabels=False, zeroline=False),
        showlegend=False,
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

def create_graph_visualization(nodes=None, edges=None, directed=False):
    """Create an interactive graph visualization"""
    if nodes is None:
        nodes = ['A', 'B', 'C', 'D', 'E']
    if edges is None:
        edges = [('A', 'B'), ('A', 'C'), ('B', 'D'), ('C', 'D'), ('C', 'E'), ('D', 'E')]
    
    # Create a graph
    G = nx.DiGraph() if directed else nx.Graph()
    
    # Add nodes
    G.add_nodes_from(nodes)
    
    # Add edges
    G.add_edges_from(edges)
    
    # Get positions for nodes using a spring layout
    pos = nx.spring_layout(G, seed=42)
    
    # Create visualization
    edge_x = []
    edge_y = []
    
    # Add edges
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines',
        showlegend=False
    )
    
    # Add nodes
    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=list(G.nodes()),
        textposition="middle center",
        marker=dict(
            size=30,
            color='lightblue',
            line=dict(width=2, color='navy')
        ),
        showlegend=False
    )
    
    # Create a figure
    fig = go.Figure(data=[edge_trace, node_trace],
                  layout=go.Layout(
                      title='Graph Visualization',
                      showlegend=False,
                      hovermode='closest',
                      margin=dict(b=20, l=5, r=5, t=40),
                      xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                      yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                  ))
    
    return fig

def create_sorting_visualization():
    """Create an interactive sorting algorithm visualization"""
    # Create sample data
    data = np.random.randint(1, 100, 10)
    
    # Create a visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create the bar chart
    bars = ax.bar(range(len(data)), data, color='lightblue', edgecolor='navy')
    
    # Set the limits
    ax.set_xlim(-0.5, len(data) - 0.5)
    ax.set_ylim(0, max(data) * 1.1)
    
    # Remove the ticks and labels
    ax.set_xticks(range(len(data)))
    ax.set_xticklabels([str(i) for i in range(len(data))])
    ax.set_title('Sorting Visualization')
    
    # Convert to base64 for display
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    
    return f"data:image/png;base64,{img_str}"

# System Design Visualizations

def create_scalability_visualization():
    """Create an interactive scalability visualization"""
    # Create sample data
    users = [500, 1000, 5000, 10000, 50000, 100000]
    response_times_vertical = [10, 18, 35, 70, 250, 500]  # Vertical scaling
    response_times_horizontal = [10, 12, 16, 20, 30, 45]  # Horizontal scaling
    
    # Create figure
    fig = go.Figure()
    
    # Add vertical scaling line
    fig.add_trace(go.Scatter(
        x=users,
        y=response_times_vertical,
        mode='lines+markers',
        name='Vertical Scaling',
        line=dict(color='red', width=2),
        marker=dict(size=8)
    ))
    
    # Add horizontal scaling line
    fig.add_trace(go.Scatter(
        x=users,
        y=response_times_horizontal,
        mode='lines+markers',
        name='Horizontal Scaling',
        line=dict(color='green', width=2),
        marker=dict(size=8)
    ))
    
    # Update layout
    fig.update_layout(
        title='Scaling Performance Comparison',
        xaxis_title='Number of Users',
        yaxis_title='Response Time (ms)',
        xaxis=dict(type='log'),
        height=400,
        margin=dict(l=50, r=20, t=50, b=50)
    )
    
    return fig

def create_load_balancing_visualization():
    """Create an interactive load balancing visualization"""
    # Create visualization
    fig = go.Figure()
    
    # Add client
    fig.add_trace(go.Scatter(
        x=[0],
        y=[2],
        mode='markers+text',
        marker=dict(size=60, color='lightblue', symbol='square', line=dict(color='navy', width=2)),
        text=['Client'],
        textposition="middle center",
        name='Client'
    ))
    
    # Add load balancer
    fig.add_trace(go.Scatter(
        x=[2],
        y=[2],
        mode='markers+text',
        marker=dict(size=80, color='lightgreen', symbol='diamond', line=dict(color='darkgreen', width=2)),
        text=['Load<br>Balancer'],
        textposition="middle center",
        name='Load Balancer'
    ))
    
    # Add servers
    server_y = [0, 1, 3, 4]
    for i, y in enumerate(server_y):
        fig.add_trace(go.Scatter(
            x=[4],
            y=[y],
            mode='markers+text',
            marker=dict(size=60, color='lightyellow', symbol='square', line=dict(color='orange', width=2)),
            text=[f'Server {i+1}'],
            textposition="middle center",
            name=f'Server {i+1}'
        ))
    
    # Add arrows
    # Client to load balancer
    fig.add_annotation(
        x=1,
        y=2,
        ax=0.2,
        ay=2,
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        showarrow=True,
        arrowhead=2,
        arrowsize=1.5,
        arrowwidth=2,
        arrowcolor='navy'
    )
    
    # Load balancer to servers
    colors = ['red', 'blue', 'green', 'purple']
    for i, y in enumerate(server_y):
        fig.add_annotation(
            x=3,
            y=y,
            ax=2.2,
            ay=2,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=2,
            arrowsize=1.5,
            arrowwidth=2,
            arrowcolor=colors[i]
        )
    
    # Update layout
    fig.update_layout(
        title='Load Balancing Architecture',
        xaxis=dict(range=[-1, 5], showticklabels=False, zeroline=False),
        yaxis=dict(range=[-1, 5], showticklabels=False, zeroline=False),
        showlegend=False,
        height=400,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig

def create_caching_visualization():
    """Create an interactive caching visualization"""
    # Create sample data
    cache_sizes = [0, 10, 20, 50, 100, 200]  # MB
    response_times = [120, 85, 60, 35, 22, 20]  # ms
    hit_rates = [0, 30, 50, 75, 90, 95]  # percentage
    
    # Create figure with secondary y-axis
    fig = go.Figure()
    
    # Add response time line
    fig.add_trace(go.Scatter(
        x=cache_sizes,
        y=response_times,
        mode='lines+markers',
        name='Response Time (ms)',
        line=dict(color='blue', width=2),
        marker=dict(size=8)
    ))
    
    # Add hit rate line
    fig.add_trace(go.Scatter(
        x=cache_sizes,
        y=hit_rates,
        mode='lines+markers',
        name='Cache Hit Rate (%)',
        line=dict(color='green', width=2),
        marker=dict(size=8),
        yaxis='y2'
    ))
    
    # Update layout with secondary y-axis
    fig.update_layout(
        title='Effect of Cache Size on Performance',
        xaxis_title='Cache Size (MB)',
        yaxis_title='Response Time (ms)',
        yaxis2=dict(
            title='Cache Hit Rate (%)',
            titlefont=dict(color='green'),
            tickfont=dict(color='green'),
            overlaying='y',
            side='right'
        ),
        height=400,
        margin=dict(l=50, r=50, t=50, b=50),
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)')
    )
    
    return fig

def create_database_design_visualization():
    """Create a database architecture visualization"""
    # Create visualization
    fig = go.Figure()
    
    # Add primary database
    fig.add_trace(go.Scatter(
        x=[2],
        y=[3],
        mode='markers+text',
        marker=dict(size=100, color='lightblue', symbol='cylinder', line=dict(color='navy', width=2)),
        text=['Primary<br>Database'],
        textposition="middle center",
        name='Primary'
    ))
    
    # Add read replicas
    for i, pos in enumerate([(0.5, 1), (2, 1), (3.5, 1)]):
        fig.add_trace(go.Scatter(
            x=[pos[0]],
            y=[pos[1]],
            mode='markers+text',
            marker=dict(size=80, color='lightgreen', symbol='cylinder', line=dict(color='darkgreen', width=2)),
            text=[f'Read<br>Replica {i+1}'],
            textposition="middle center",
            name=f'Replica {i+1}'
        ))
    
    # Add arrows from primary to replicas
    for i, pos in enumerate([(0.5, 1), (2, 1), (3.5, 1)]):
        fig.add_annotation(
            x=pos[0],
            y=pos[1] + 0.5,
            ax=2,
            ay=3 - 0.5,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=2,
            arrowsize=1.5,
            arrowwidth=2,
            arrowcolor='navy',
            text="Replication"
        )
    
    # Add application servers
    for i, x in enumerate([1, 3]):
        fig.add_trace(go.Scatter(
            x=[x],
            y=[5],
            mode='markers+text',
            marker=dict(size=80, color='#FFD580', symbol='square', line=dict(color='orange', width=2)),
            text=[f'App<br>Server {i+1}'],
            textposition="middle center",
            name=f'App Server {i+1}'
        ))
        
        # Add write arrows to primary
        fig.add_annotation(
            x=2,
            y=3 + 0.5,
            ax=x,
            ay=5 - 0.5,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=2,
            arrowsize=1.5,
            arrowwidth=2,
            arrowcolor='red',
            text="Writes"
        )
        
        # Add read arrows to replicas
        for j, pos in enumerate([(0.5, 1), (2, 1), (3.5, 1)]):
            if (i == 0 and j < 2) or (i == 1 and j >= 1):  # Distribute reads
                fig.add_annotation(
                    x=pos[0],
                    y=1 + 0.5,
                    ax=x,
                    ay=5 - 0.5,
                    xref="x",
                    yref="y",
                    axref="x",
                    ayref="y",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1.5,
                    arrowwidth=2,
                    arrowcolor='green',
                    text="Reads"
                )
    
    # Update layout
    fig.update_layout(
        title='Database Replication Architecture',
        xaxis=dict(range=[-1, 5], showticklabels=False, zeroline=False),
        yaxis=dict(range=[0, 6], showticklabels=False, zeroline=False),
        showlegend=False,
        height=500,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig

def create_microservices_visualization():
    """Create a microservices architecture visualization"""
    # Create visualization
    fig = go.Figure()
    
    # Add API Gateway
    fig.add_trace(go.Scatter(
        x=[2],
        y=[5],
        mode='markers+text',
        marker=dict(size=100, color='lightblue', symbol='square', line=dict(color='navy', width=2)),
        text=['API<br>Gateway'],
        textposition="middle center",
        name='API Gateway'
    ))
    
    # Add Microservices
    services = [
        ('User Service', 0, 3),
        ('Product Service', 2, 3),
        ('Order Service', 4, 3),
        ('Payment Service', 1, 1),
        ('Notification Service', 3, 1)
    ]
    
    for name, x, y in services:
        fig.add_trace(go.Scatter(
            x=[x],
            y=[y],
            mode='markers+text',
            marker=dict(size=80, color='lightgreen', symbol='square', line=dict(color='darkgreen', width=2)),
            text=[name],
            textposition="middle center",
            name=name
        ))
        
        # Add arrows from API Gateway to services
        if y == 3:  # First layer services
            fig.add_annotation(
                x=x,
                y=y + 0.5,
                ax=2,
                ay=5 - 0.5,
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1.5,
                arrowwidth=2,
                arrowcolor='navy'
            )
    
    # Add arrows between services
    connections = [
        (0, 3, 1, 1), # User Service to Payment Service
        (2, 3, 3, 1), # Product Service to Notification Service
        (4, 3, 3, 1), # Order Service to Notification Service
        (4, 3, 1, 1)  # Order Service to Payment Service
    ]
    
    for x1, y1, x2, y2 in connections:
        fig.add_annotation(
            x=x2,
            y=y2 + 0.5,
            ax=x1,
            ay=y1 - 0.5,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=2,
            arrowsize=1.5,
            arrowwidth=2,
            arrowcolor='green'
        )
    
    # Add databases
    databases = [
        ('User DB', 0, 0),
        ('Product DB', 2, 0),
        ('Order DB', 4, 0)
    ]
    
    for name, x, y in databases:
        fig.add_trace(go.Scatter(
            x=[x],
            y=[y],
            mode='markers+text',
            marker=dict(size=70, color='#FFD580', symbol='cylinder', line=dict(color='orange', width=2)),
            text=[name],
            textposition="middle center",
            name=name
        ))
        
        # Add arrows from services to databases
        fig.add_annotation(
            x=x,
            y=y + 0.5,
            ax=x,
            ay=1 if name == 'User DB' or name == 'Product DB' else 3 - 0.5,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=2,
            arrowsize=1.5,
            arrowwidth=2,
            arrowcolor='purple'
        )
    
    # Update layout
    fig.update_layout(
        title='Microservices Architecture',
        xaxis=dict(range=[-1, 5], showticklabels=False, zeroline=False),
        yaxis=dict(range=[-1, 6], showticklabels=False, zeroline=False),
        showlegend=False,
        height=600,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig

def create_distributed_systems_visualization():
    """Create a distributed systems architecture visualization"""
    # Create a graph for distributed system
    G = nx.Graph()
    
    # Add nodes with types
    nodes = [
        ('Load Balancer', 'gateway'),
        ('App Server 1', 'server'),
        ('App Server 2', 'server'),
        ('App Server 3', 'server'),
        ('Cache 1', 'cache'),
        ('Cache 2', 'cache'),
        ('Database 1', 'db'),
        ('Database 2', 'db'),
        ('Message Queue', 'queue'),
        ('Worker 1', 'worker'),
        ('Worker 2', 'worker')
    ]
    
    for name, node_type in nodes:
        G.add_node(name, type=node_type)
    
    # Add edges
    edges = [
        ('Load Balancer', 'App Server 1'),
        ('Load Balancer', 'App Server 2'),
        ('Load Balancer', 'App Server 3'),
        ('App Server 1', 'Cache 1'),
        ('App Server 2', 'Cache 1'),
        ('App Server 3', 'Cache 2'),
        ('App Server 1', 'Database 1'),
        ('App Server 2', 'Database 1'),
        ('App Server 3', 'Database 2'),
        ('App Server 1', 'Message Queue'),
        ('App Server 2', 'Message Queue'),
        ('App Server 3', 'Message Queue'),
        ('Message Queue', 'Worker 1'),
        ('Message Queue', 'Worker 2'),
        ('Worker 1', 'Database 1'),
        ('Worker 2', 'Database 2')
    ]
    
    G.add_edges_from(edges)
    
    # Get positions for nodes
    pos = {
        'Load Balancer': (0, 2),
        'App Server 1': (2, 3),
        'App Server 2': (2, 2),
        'App Server 3': (2, 1),
        'Cache 1': (4, 3),
        'Cache 2': (4, 1),
        'Database 1': (6, 3),
        'Database 2': (6, 1),
        'Message Queue': (4, 2),
        'Worker 1': (6, 2.5),
        'Worker 2': (6, 1.5)
    }
    
    # Set node colors based on type
    color_map = {
        'gateway': 'lightblue',
        'server': 'lightgreen',
        'cache': 'lightyellow',
        'db': 'lightpink',
        'queue': 'lightgrey',
        'worker': 'lavender'
    }
    
    node_colors = [color_map[G.nodes[node]['type']] for node in G.nodes()]
    
    # Create edges for plot
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Create nodes for plot
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        node_color.append(color_map[G.nodes[node]['type']])
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        textposition="middle center",
        marker=dict(
            size=30,
            color=node_color,
            line=dict(width=2, color='navy')
        )
    )
    
    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace],
                  layout=go.Layout(
                      title='Distributed System Architecture',
                      showlegend=False,
                      hovermode='closest',
                      margin=dict(b=20, l=5, r=5, t=40),
                      xaxis=dict(range=[-1, 7], showgrid=False, zeroline=False, showticklabels=False),
                      yaxis=dict(range=[0, 4], showgrid=False, zeroline=False, showticklabels=False),
                      height=500
                  ))
    
    return fig

# Math Visualizations

def create_infinite_series_visualization():
    """Create an interactive infinite series visualization"""
    # Create data for partial sums
    n_terms = list(range(1, 21))
    
    # Create figure
    fig = go.Figure()
    
    # Add trace for harmonic series
    harmonic_series = [sum(1/i for i in range(1, n+1)) for n in n_terms]
    fig.add_trace(go.Scatter(
        x=n_terms,
        y=harmonic_series,
        mode='lines+markers',
        name='Harmonic Series',
        text=[f'Sum of {n} terms: {val:.4f}' for n, val in zip(n_terms, harmonic_series)],
        line=dict(color='blue', width=2),
        marker=dict(size=8)
    ))
    
    # Add trace for p-series with p=2 (sum of 1/n²)
    p_series = [sum(1/(i**2) for i in range(1, n+1)) for n in n_terms]
    fig.add_trace(go.Scatter(
        x=n_terms,
        y=p_series,
        mode='lines+markers',
        name='p-Series (p=2)',
        text=[f'Sum of {n} terms: {val:.4f}' for n, val in zip(n_terms, p_series)],
        line=dict(color='red', width=2),
        marker=dict(size=8)
    ))
    
    # Add trace for geometric series with r=1/2
    geometric_series = [sum((1/2)**i for i in range(n)) for n in n_terms]
    fig.add_trace(go.Scatter(
        x=n_terms,
        y=geometric_series,
        mode='lines+markers',
        name='Geometric Series (r=1/2)',
        text=[f'Sum of {n} terms: {val:.4f}' for n, val in zip(n_terms, geometric_series)],
        line=dict(color='green', width=2),
        marker=dict(size=8)
    ))
    
    # Add line for limit of geometric series
    fig.add_shape(
        type="line",
        x0=0, y0=2, x1=20, y1=2,
        line=dict(
            color="green",
            width=1,
            dash="dash",
        )
    )
    
    fig.add_annotation(
        x=18, y=2.1,
        text="Limit = 2",
        showarrow=False,
        font=dict(color="green")
    )
    
    # Add line for limit of p-series
    fig.add_shape(
        type="line",
        x0=0, y0=1.645, x1=20, y1=1.645,
        line=dict(
            color="red",
            width=1,
            dash="dash",
        )
    )
    
    fig.add_annotation(
        x=18, y=1.7,
        text="Limit = π²/6",
        showarrow=False,
        font=dict(color="red")
    )
    
    # Update layout
    fig.update_layout(
        title='Partial Sums of Various Infinite Series',
        xaxis_title='Number of Terms (n)',
        yaxis_title='Sum Value',
        yaxis=dict(range=[0, 4]),
        height=500,
        margin=dict(l=50, r=20, t=50, b=50),
        legend=dict(x=0.01, y=0.99)
    )
    
    return fig

def create_eigenvalues_visualization():
    """Create an interactive eigenvalues and eigenvectors visualization"""
    # Create a sample 2x2 matrix with real eigenvalues
    A = np.array([[3, 1], [1, 2]])
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    # Generate points for visualization
    x = np.linspace(-3, 3, 20)
    y = np.linspace(-3, 3, 20)
    X, Y = np.meshgrid(x, y)
    
    # Transform points using matrix A
    positions = np.vstack([X.ravel(), Y.ravel()])
    transformed = A @ positions
    
    # Reshape back to grid
    X_transformed = transformed[0, :].reshape(X.shape)
    Y_transformed = transformed[1, :].reshape(Y.shape)
    
    # Create visualization
    fig = go.Figure()
    
    # Add vector field
    fig.add_trace(go.Scatter(
        x=positions[0, :],
        y=positions[1, :],
        mode='markers',
        marker=dict(
            size=3,
            color='lightblue'
        ),
        name='Original Points'
    ))
    
    # Add transformed points
    fig.add_trace(go.Scatter(
        x=transformed[0, :],
        y=transformed[1, :],
        mode='markers',
        marker=dict(
            size=3,
            color='red'
        ),
        name='Transformed Points'
    ))
    
    # Add eigenvectors
    for i in range(len(eigenvalues)):
        v = eigenvectors[:, i]
        # Scale eigenvector for better visualization
        scale = 2.5
        fig.add_trace(go.Scatter(
            x=[0, v[0] * scale],
            y=[0, v[1] * scale],
            mode='lines',
            line=dict(
                color='green',
                width=3
            ),
            name=f'Eigenvector {i+1} (λ={eigenvalues[i]:.2f})'
        ))
        
        # Add transformed eigenvector which should be parallel to original
        v_transformed = A @ v
        fig.add_trace(go.Scatter(
            x=[0, v_transformed[0]],
            y=[0, v_transformed[1]],
            mode='lines',
            line=dict(
                color='purple',
                width=3,
                dash='dash'
            ),
            name=f'Transformed Eigenvector {i+1}'
        ))
    
    # Update layout
    fig.update_layout(
        title='Eigenvalues and Eigenvectors Visualization',
        xaxis=dict(
            range=[-3, 3],
            zeroline=True,
            showgrid=True,
            title='x'
        ),
        yaxis=dict(
            range=[-3, 3],
            zeroline=True,
            showgrid=True,
            title='y',
            scaleanchor="x",
            scaleratio=1
        ),
        height=600,
        margin=dict(l=50, r=20, t=50, b=50),
        legend=dict(x=0.01, y=0.99)
    )
    
    return fig

def create_probability_visualization():
    """Create an interactive probability visualization"""
    # Create normal distribution visualization
    fig = go.Figure()
    
    # Generate normal distribution data
    x = np.linspace(-4, 4, 1000)
    y_normal = 1 / np.sqrt(2 * np.pi) * np.exp(-x**2 / 2)
    
    # Add normal distribution curve
    fig.add_trace(go.Scatter(
        x=x,
        y=y_normal,
        mode='lines',
        name='Standard Normal Distribution',
        line=dict(color='blue', width=2)
    ))
    
    # Add area for standard deviation ranges
    x_fill = np.linspace(-1, 1, 100)
    y_fill = 1 / np.sqrt(2 * np.pi) * np.exp(-x_fill**2 / 2)
    
    fig.add_trace(go.Scatter(
        x=np.concatenate([x_fill, x_fill[::-1]]),
        y=np.concatenate([y_fill, np.zeros_like(y_fill)]),
        fill='toself',
        fillcolor='rgba(0, 0, 255, 0.2)',
        line=dict(color='rgba(255, 255, 255, 0)'),
        name='Within 1σ (68.2%)'
    ))
    
    x_fill = np.linspace(-2, 2, 100)
    y_fill = 1 / np.sqrt(2 * np.pi) * np.exp(-x_fill**2 / 2)
    
    fig.add_trace(go.Scatter(
        x=np.concatenate([x_fill, x_fill[::-1]]),
        y=np.concatenate([y_fill, np.zeros_like(y_fill)]),
        fill='toself',
        fillcolor='rgba(0, 255, 0, 0.1)',
        line=dict(color='rgba(255, 255, 255, 0)'),
        name='Within 2σ (95.4%)'
    ))
    
    x_fill = np.linspace(-3, 3, 100)
    y_fill = 1 / np.sqrt(2 * np.pi) * np.exp(-x_fill**2 / 2)
    
    fig.add_trace(go.Scatter(
        x=np.concatenate([x_fill, x_fill[::-1]]),
        y=np.concatenate([y_fill, np.zeros_like(y_fill)]),
        fill='toself',
        fillcolor='rgba(255, 0, 0, 0.05)',
        line=dict(color='rgba(255, 255, 255, 0)'),
        name='Within 3σ (99.7%)'
    ))
    
    # Add annotations for standard deviations
    for i, color in zip(range(1, 4), ['blue', 'green', 'red']):
        fig.add_annotation(
            x=i,
            y=0.01,
            text=f'{i}σ',
            showarrow=False,
            font=dict(color=color)
        )
        fig.add_annotation(
            x=-i,
            y=0.01,
            text=f'-{i}σ',
            showarrow=False,
            font=dict(color=color)
        )
    
    # Update layout
    fig.update_layout(
        title='Normal Distribution and Probability',
        xaxis_title='Standard Deviations from Mean (σ)',
        yaxis_title='Probability Density',
        height=500,
        margin=dict(l=50, r=20, t=50, b=50),
        legend=dict(x=0.01, y=0.99)
    )
    
    return fig

def create_regression_visualization():
    """Create an interactive regression analysis visualization"""
    # Generate some sample data
    np.random.seed(42)
    x = np.linspace(0, 10, 50)
    y = 2 * x + 1 + np.random.normal(0, 1.5, size=len(x))
    
    # Fit linear regression
    coeffs = np.polyfit(x, y, 1)
    poly_line = np.poly1d(coeffs)
    
    # Calculate R-squared
    y_hat = poly_line(x)
    y_bar = np.mean(y)
    ssreg = np.sum((y_hat - y_bar) ** 2)
    sstot = np.sum((y - y_bar) ** 2)
    r_squared = ssreg / sstot
    
    # Create scatter plot with regression line
    fig = go.Figure()
    
    # Add scatter plot of data points
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='markers',
        name='Data Points',
        marker=dict(
            size=8,
            color='blue',
            opacity=0.7
        )
    ))
    
    # Add regression line
    fig.add_trace(go.Scatter(
        x=x,
        y=poly_line(x),
        mode='lines',
        name=f'Regression Line (y = {coeffs[0]:.2f}x + {coeffs[1]:.2f})',
        line=dict(
            color='red',
            width=2
        )
    ))
    
    # Update layout
    fig.update_layout(
        title=f'Linear Regression (R² = {r_squared:.3f})',
        xaxis_title='x',
        yaxis_title='y',
        height=500,
        margin=dict(l=50, r=20, t=50, b=50),
        legend=dict(x=0.01, y=0.99)
    )
    
    return fig

def create_combinatorics_visualization():
    """Create an interactive combinatorics visualization"""
    # Generate data for Pascal's triangle
    def pascal_triangle(n):
        triangle = []
        for i in range(n):
            row = [1]
            for j in range(1, i):
                row.append(triangle[i-1][j-1] + triangle[i-1][j])
            if i > 0:
                row.append(1)
            triangle.append(row)
        return triangle
    
    # Create Pascal's triangle with 10 rows
    triangle = pascal_triangle(10)
    
    # Create visualization
    fig = go.Figure()
    
    # Add cells for each value in Pascal's triangle
    for i, row in enumerate(triangle):
        x_positions = [j - len(row)/2 + 0.5 for j in range(len(row))]
        
        for j, value in enumerate(row):
            fig.add_trace(go.Scatter(
                x=[x_positions[j]],
                y=[-i],  # Negative to display top-to-bottom
                mode='markers+text',
                marker=dict(
                    size=40,
                    color=f'rgba(0, 0, 255, {min(value/20, 0.8)})',
                    line=dict(color='navy', width=1)
                ),
                text=[str(value)],
                textposition="middle center",
                textfont=dict(size=12, color='white' if value > 10 else 'black'),
                showlegend=False
            ))
    
    # Add title to indicate formula
    fig.add_annotation(
        x=0,
        y=1,
        xref="paper",
        yref="paper",
        text="Pascal's Triangle shows binomial coefficients C(n,k)",
        showarrow=False,
        font=dict(size=14)
    )
    
    # Add formula annotation
    fig.add_annotation(
        x=0.5,
        y=0.95,
        xref="paper",
        yref="paper",
        text="C(n,k) = n! / [k! × (n-k)!]",
        showarrow=False,
        font=dict(size=14)
    )
    
    # Update layout
    fig.update_layout(
        title="Combinatorics: Pascal's Triangle",
        xaxis=dict(range=[-5, 5], showticklabels=False, zeroline=False),
        yaxis=dict(range=[-10, 1], showticklabels=False, zeroline=False),
        height=600,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig

def create_graph_theory_visualization():
    """Create an interactive graph theory visualization"""
    # Create several classic graph types
    graphs = {
        "Complete": nx.complete_graph(5),
        "Cycle": nx.cycle_graph(8),
        "Star": nx.star_graph(7),
        "Path": nx.path_graph(6),
        "Wheel": nx.wheel_graph(8),
        "Grid": nx.grid_2d_graph(3, 3)
    }
    
    # Convert 2D grid positions to 1D names
    if "Grid" in graphs:
        mapping = {pos: f"{pos[0]},{pos[1]}" for pos in graphs["Grid"].nodes()}
        graphs["Grid"] = nx.relabel_nodes(graphs["Grid"], mapping)
    
    # Fixed positions for each graph type
    positions = {
        "Complete": nx.circular_layout(graphs["Complete"]),
        "Cycle": nx.circular_layout(graphs["Cycle"]),
        "Star": nx.spring_layout(graphs["Star"], seed=42),
        "Path": {i: (i, 0) for i in range(6)},
        "Wheel": nx.spring_layout(graphs["Wheel"], seed=42),
        "Grid": {f"{i},{j}": (i, j) for i in range(3) for j in range(3)}
    }
    
    # Create subplots
    fig = go.Figure()
    
    # Colors for each graph
    colors = {
        "Complete": "red",
        "Cycle": "blue",
        "Star": "green",
        "Path": "purple",
        "Wheel": "orange",
        "Grid": "teal"
    }
    
    # Create a scatter plot for each graph
    for graph_name, G in graphs.items():
        # Get positions
        pos = positions[graph_name]
        
        # Create edges
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        # Create nodes
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        
        # Shift the graph to its subplot position
        shift_x = {"Complete": -4, "Cycle": 0, "Star": 4, "Path": -4, "Wheel": 0, "Grid": 4}
        shift_y = {"Complete": 4, "Cycle": 4, "Star": 4, "Path": 0, "Wheel": 0, "Grid": 0}
        
        edge_x = [x + shift_x[graph_name] if x is not None else None for x in edge_x]
        edge_y = [y + shift_y[graph_name] if y is not None else None for y in edge_y]
        node_x = [x + shift_x[graph_name] for x in node_x]
        node_y = [y + shift_y[graph_name] for y in node_y]
        
        # Add edges
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1.5, color=colors[graph_name]),
            hoverinfo='none',
            mode='lines',
            showlegend=False
        ))
        
        # Add nodes
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            marker=dict(
                size=15,
                color='white',
                line=dict(width=2, color=colors[graph_name])
            ),
            name=graph_name
        ))
        
        # Add label
        fig.add_annotation(
            x=shift_x[graph_name],
            y=shift_y[graph_name] - 2,
            text=graph_name,
            showarrow=False,
            font=dict(size=14, color=colors[graph_name])
        )
    
    # Update layout
    fig.update_layout(
        title='Common Graph Structures in Graph Theory',
        showlegend=False,
        xaxis=dict(range=[-6, 6], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[-4, 6], showgrid=False, zeroline=False, showticklabels=False),
        height=600,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig

# Main visualization selector function
def get_visualization(topic_id, category=None):
    """Return the appropriate visualization for a given topic"""
    # DSA visualizations
    if topic_id == "arrays":
        return create_array_visualization()
    elif topic_id == "linked_lists":
        return None  # Already implemented in main app
    elif topic_id == "stacks":
        return create_stack_visualization()
    elif topic_id == "queues":
        return create_queue_visualization()
    elif topic_id == "hash_tables":
        return create_hash_table_visualization()
    elif topic_id == "trees" or topic_id == "binary_search_trees":
        return create_binary_tree_visualization()
    elif topic_id == "heaps":
        return create_binary_tree_visualization()  # Similar to binary tree but with heap property
    elif topic_id == "graphs":
        return create_graph_visualization()
    elif topic_id == "sorting":
        return create_sorting_visualization()
    elif topic_id == "searching":
        return create_array_visualization()  # Show array with search indicators
    
    # System Design visualizations
    elif topic_id == "scalability":
        return create_scalability_visualization()
    elif topic_id == "load_balancing":
        return create_load_balancing_visualization()
    elif topic_id == "caching":
        return create_caching_visualization()
    elif topic_id == "database_design":
        return create_database_design_visualization()
    elif topic_id == "microservices":
        return create_microservices_visualization()
    elif topic_id == "distributed_systems":
        return create_distributed_systems_visualization()
    
    # Math visualizations
    elif topic_id == "calculus_series":
        return create_infinite_series_visualization()
    elif topic_id == "linear_algebra_eigen":
        return create_eigenvalues_visualization()
    elif topic_id == "statistics_probability":
        return create_probability_visualization()
    elif topic_id == "statistics_regression":
        return create_regression_visualization()
    elif topic_id == "discrete_math_combinatorics":
        return create_combinatorics_visualization()
    elif topic_id == "discrete_math_graph_theory":
        return create_graph_theory_visualization()
    
    # Fallback to None if no specific visualization
    return None