"""
Archimedes Chess AI - Interactive Dashboard
Streamlit-based dashboard with live metrics, visualization, and play vs AI.
"""


import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import sqlite3
import logging
import chess
import chess.svg
import torch
import time
import json
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional
import base64
import os
from io import BytesIO

# Import project modules
from model import ChessResNet, AlphaZeroEncoder
from mcts import MCTS
from metrics import MetricsLogger
from utils import safe_load_checkpoint, setup_logging

logger = logging.getLogger(__name__)

class ReplayBuffer:
    def __init__(self, *args, **kwargs):
        pass

try:
    sys.modules.get("__main__").ReplayBuffer = ReplayBuffer
except Exception:
    pass


# Page configuration
st.set_page_config(
    page_title="Archimedes Chess AI Dashboard",
    page_icon="♟️",
    layout="wide",
    initial_sidebar_state="expanded"
)


class DashboardData:
    """Handles data loading from metrics database."""
    
    def __init__(self, db_path: str = "logs/training_logs.db"):
        self.db_path = db_path

    def _safe_query(self, query: str, params: tuple = ()) -> pd.DataFrame:
        """Execute a query safely and return a DataFrame."""
        if not Path(self.db_path).exists():
             return pd.DataFrame()

        try:
            with sqlite3.connect(self.db_path) as conn:
                return pd.read_sql_query(query, conn, params=params)
        except sqlite3.OperationalError as e:
            # Handle "no such table" errors gracefully for fresh installs
            if "no such table" in str(e):
                return pd.DataFrame()
            st.error(f"Database error: {e}")
            return pd.DataFrame()
        except Exception as e:
            st.error(f"Database error: {e}")
            return pd.DataFrame()
    
    def get_training_metrics(self, limit: int = 1000) -> pd.DataFrame:
        """Load training metrics."""
        query = "SELECT * FROM training_metrics ORDER BY timestamp DESC LIMIT ?"
        return self._safe_query(query, (limit,))
    
    def get_mcts_metrics(self, limit: int = 1000) -> pd.DataFrame:
        """Load MCTS metrics."""
        query = "SELECT * FROM mcts_metrics ORDER BY timestamp DESC LIMIT ?"
        return self._safe_query(query, (limit,))
    
    def get_chess_metrics(self, limit: int = 1000) -> pd.DataFrame:
        """Load chess-specific metrics."""
        query = "SELECT * FROM chess_metrics ORDER BY timestamp DESC LIMIT ?"
        return self._safe_query(query, (limit,))
    
    def get_hardware_metrics(self, limit: int = 1000) -> pd.DataFrame:
        """Load hardware metrics."""
        query = "SELECT * FROM hardware_metrics ORDER BY timestamp DESC LIMIT ?"
        return self._safe_query(query, (limit,))
    
    def get_games(self, limit: int = 100) -> pd.DataFrame:
        """Load game records."""
        query = "SELECT * FROM games ORDER BY timestamp DESC LIMIT ?"
        return self._safe_query(query, (limit,))
    
    def get_position_analysis(self, limit: int = 100) -> pd.DataFrame:
        """Load position analysis data."""
        query = "SELECT * FROM position_analysis ORDER BY timestamp DESC LIMIT ?"
        return self._safe_query(query, (limit,))


@st.cache_resource
def load_model(checkpoint_path: str = "checkpoints/latest.pt"):
    """Load the trained model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        model = ChessResNet()
        encoder = AlphaZeroEncoder()

        if Path(checkpoint_path).exists():
            checkpoint = safe_load_checkpoint(checkpoint_path, device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            model.eval()
            return model, encoder, device
        else:
            st.warning(f"Checkpoint not found: {checkpoint_path}")
            return None, None, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, device


def resolve_checkpoint_path(requested_path: str) -> str:
    requested = Path(requested_path)
    if requested.exists():
        return str(requested)

    checkpoints_dir = Path("./checkpoints").resolve()
    preferred = [
        checkpoints_dir / "latest_model.pt",
        checkpoints_dir / "latest.pt",
        checkpoints_dir / "latest_checkpoint.pt",
    ]
    for p in preferred:
        if p.exists():
            return str(p)

    candidates = list(checkpoints_dir.glob("*.pt")) + list(checkpoints_dir.glob("*.pth"))
    if not candidates:
        return requested_path

    newest = max(candidates, key=lambda p: p.stat().st_mtime)
    return str(newest)


def render_board_svg(board: chess.Board, size: int = 400) -> str:
    """Render chess board as SVG."""
    svg = chess.svg.board(board, size=size)
    return svg


def plot_training_loss(df: pd.DataFrame):
    """Plot training loss over time."""
    if df.empty:
        st.info("No training data available yet")
        return
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Total Loss", "Policy Loss", "Value Loss", "Learning Rate"),
        vertical_spacing=0.12
    )
    
    # Total loss
    fig.add_trace(
        go.Scatter(x=df['epoch'], y=df['loss_total'], mode='lines', name='Total Loss',
                  line=dict(color='#FF6B6B', width=2)),
        row=1, col=1
    )
    
    # Policy loss
    fig.add_trace(
        go.Scatter(x=df['epoch'], y=df['loss_policy'], mode='lines', name='Policy Loss',
                  line=dict(color='#4ECDC4', width=2)),
        row=1, col=2
    )
    
    # Value loss
    fig.add_trace(
        go.Scatter(x=df['epoch'], y=df['loss_value'], mode='lines', name='Value Loss',
                  line=dict(color='#95E1D3', width=2)),
        row=2, col=1
    )
    
    # Learning rate
    fig.add_trace(
        go.Scatter(x=df['epoch'], y=df['learning_rate'], mode='lines', name='Learning Rate',
                  line=dict(color='#F38181', width=2)),
        row=2, col=2
    )
    
    fig.update_xaxes(title_text="Epoch")
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="Loss", row=1, col=2)
    fig.update_yaxes(title_text="Loss", row=2, col=1)
    fig.update_yaxes(title_text="Rate", row=2, col=2)
    
    fig.update_layout(height=600, showlegend=False, template="plotly_dark")
    
    st.plotly_chart(fig, use_container_width=True)


def plot_accuracy(df: pd.DataFrame):
    """Plot accuracy metrics."""
    if df.empty or 'accuracy_top1' not in df.columns:
        st.info("No accuracy data available yet")
        return
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['epoch'], y=df['accuracy_top1'] * 100,
        mode='lines+markers', name='Top-1 Accuracy',
        line=dict(color='#00D9FF', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=df['epoch'], y=df['accuracy_top5'] * 100,
        mode='lines+markers', name='Top-5 Accuracy',
        line=dict(color='#7B68EE', width=3)
    ))
    
    fig.update_layout(
        title="Move Prediction Accuracy",
        xaxis_title="Epoch",
        yaxis_title="Accuracy (%)",
        template="plotly_dark",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)


def plot_mcts_performance(df: pd.DataFrame):
    """Plot MCTS performance metrics."""
    if df.empty:
        st.info("No MCTS data available yet")
        return
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Search Depth", "Nodes per Second", "Cache Hit Rate", "Q-Value Distribution"),
        vertical_spacing=0.12
    )
    
    # Search depth
    fig.add_trace(
        go.Scatter(x=df['epoch'], y=df['avg_search_depth'], mode='lines', name='Avg Depth',
                  line=dict(color='#FFD93D', width=2)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['epoch'], y=df['max_search_depth'], mode='lines', name='Max Depth',
                  line=dict(color='#FF6B9D', width=2)),
        row=1, col=1
    )
    
    # Nodes per second
    fig.add_trace(
        go.Scatter(x=df['epoch'], y=df['nodes_per_second'], mode='lines', name='NPS',
                  line=dict(color='#6BCB77', width=2)),
        row=1, col=2
    )
    
    # Cache hit rate
    fig.add_trace(
        go.Scatter(x=df['epoch'], y=df['cache_hit_rate'] * 100, mode='lines', name='Hit Rate',
                  line=dict(color='#4D96FF', width=2)),
        row=2, col=1
    )
    
    # Q-value distribution
    fig.add_trace(
        go.Scatter(x=df['epoch'], y=df['q_value_mean'], mode='lines', name='Q Mean',
                  line=dict(color='#C780FA', width=2)),
        row=2, col=2
    )
    
    fig.update_xaxes(title_text="Epoch")
    fig.update_layout(height=600, showlegend=True, template="plotly_dark")
    
    st.plotly_chart(fig, use_container_width=True)


def plot_chess_performance(df: pd.DataFrame):
    """Plot chess-specific performance."""
    if df.empty:
        st.info("No chess performance data available yet")
        return
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Elo Rating", "Win/Loss/Draw Rates", "Game Length", "Performance by Color"),
        specs=[[{"type": "scatter"}, {"type": "bar"}],
               [{"type": "scatter"}, {"type": "bar"}]],
        vertical_spacing=0.15
    )
    
    # Elo rating
    fig.add_trace(
        go.Scatter(x=df['epoch'], y=df['elo_estimate'], mode='lines+markers',
                  name='Elo', line=dict(color='#FFD700', width=3)),
        row=1, col=1
    )
    
    # Win/Loss/Draw rates (latest)
    if not df.empty:
        latest = df.iloc[0]
        fig.add_trace(
            go.Bar(x=['Win', 'Draw', 'Loss'],
                  y=[latest['win_rate']*100, latest['draw_rate']*100, latest['loss_rate']*100],
                  marker_color=['#00D9FF', '#FFD93D', '#FF6B6B']),
            row=1, col=2
        )
    
    # Game length
    fig.add_trace(
        go.Scatter(x=df['epoch'], y=df['avg_game_length'], mode='lines+markers',
                  name='Avg Length', line=dict(color='#95E1D3', width=2)),
        row=2, col=1
    )
    
    # Performance by color (latest)
    if not df.empty:
        latest = df.iloc[0]
        fig.add_trace(
            go.Bar(x=['White', 'Black'],
                  y=[latest['white_performance']*100, latest['black_performance']*100],
                  marker_color=['#FFFFFF', '#333333']),
            row=2, col=2
        )
    
    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_xaxes(title_text="Epoch", row=2, col=1)
    fig.update_yaxes(title_text="Elo", row=1, col=1)
    fig.update_yaxes(title_text="Rate (%)", row=1, col=2)
    fig.update_yaxes(title_text="Moves", row=2, col=1)
    fig.update_yaxes(title_text="Win Rate (%)", row=2, col=2)
    
    fig.update_layout(height=700, showlegend=False, template="plotly_dark")
    
    st.plotly_chart(fig, use_container_width=True)


def plot_hardware_usage(df: pd.DataFrame):
    """Plot hardware utilization."""
    if df.empty:
        st.info("No hardware data available yet")
        return
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("GPU Utilization", "GPU Memory", "CPU & RAM", "GPU Temperature"),
        vertical_spacing=0.12
    )
    
    # GPU utilization
    fig.add_trace(
        go.Scatter(x=df['epoch'], y=df['gpu_utilization'], mode='lines',
                  name='GPU %', line=dict(color='#00FF00', width=2), fill='tozeroy'),
        row=1, col=1
    )
    
    # GPU memory
    fig.add_trace(
        go.Scatter(x=df['epoch'], y=df['gpu_memory_used'], mode='lines',
                  name='Used', line=dict(color='#FF6B6B', width=2)),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=df['epoch'], y=df['gpu_memory_total'], mode='lines',
                  name='Total', line=dict(color='#4ECDC4', width=2, dash='dash')),
        row=1, col=2
    )
    
    # CPU & RAM
    fig.add_trace(
        go.Scatter(x=df['epoch'], y=df['cpu_percent'], mode='lines',
                  name='CPU %', line=dict(color='#FFD93D', width=2)),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['epoch'], y=df['ram_used'], mode='lines',
                  name='RAM (GB)', line=dict(color='#C780FA', width=2)),
        row=2, col=1
    )
    
    # GPU temperature
    fig.add_trace(
        go.Scatter(x=df['epoch'], y=df['gpu_temperature'], mode='lines',
                  name='Temp °C', line=dict(color='#FF6B9D', width=2)),
        row=2, col=2
    )
    
    fig.update_xaxes(title_text="Epoch")
    fig.update_layout(height=600, showlegend=True, template="plotly_dark")
    
    st.plotly_chart(fig, use_container_width=True)


def play_vs_ai_tab(model, encoder, device):
    """Interactive play vs AI interface."""
    st.header("♟️ Play vs Archimedes")
    
    # Initialize game state
    if 'board' not in st.session_state:
        st.session_state.board = chess.Board()
        st.session_state.move_history = []
    
    board = st.session_state.board
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Render board
        svg = render_board_svg(board, size=500)
        st.image(svg, use_container_width=True)
    
    with col2:
        st.subheader("Game Info")
        st.write(f"**Turn:** {'White' if board.turn else 'Black'}")
        st.write(f"**Moves:** {board.fullmove_number}")
        st.write(f"**FEN:** `{board.fen()}`")
        
        if board.is_game_over():
            st.success(f"**Game Over!** Result: {board.result()}")
        
        # Move input
        st.subheader("Your Move")
        move_input = st.text_input("Enter move (e.g., e2e4):", key="move_input")
        
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            if st.button("Make Move"):
                if not move_input:
                    st.error("Please enter a move.")
                else:
                    try:
                        move = chess.Move.from_uci(move_input)
                        if move in board.legal_moves:
                            board.push(move)
                            st.session_state.move_history.append(move_input)
                            st.rerun()
                        else:
                            st.error(f"Illegal move: {move_input}")
                    except ValueError:
                        st.error(f"Invalid UCI format: {move_input}")
                    except Exception as e:
                        st.error(f"Error processing move: {e}")
        
        with col_b:
            if st.button("AI Move"):
                if model and not board.is_game_over():
                    with st.spinner("AI thinking..."):
                        mcts = MCTS(model, encoder, num_simulations=400)
                        ai_move, stats = mcts.search(board, add_noise=False)
                        board.push(ai_move)
                        st.session_state.move_history.append(ai_move.uci())
                        st.rerun()
        
        with col_c:
            if st.button("Reset"):
                st.session_state.board = chess.Board()
                st.session_state.move_history = []
                st.rerun()
        
        # Move history
        if st.session_state.move_history:
            st.subheader("Move History")
            st.write(" ".join(st.session_state.move_history))


class SafePathValidator:
    """Validates file paths against LFI attacks with strict whitelisting."""

    ALLOWED_DIRS = {
        "project_root": Path("./").resolve(),
        "checkpoints": Path("./checkpoints").resolve(),
        "logs": Path("./logs").resolve(),
    }

    # Add Colab content directory if running in Colab
    if os.path.exists("/content"):
        ALLOWED_DIRS["colab_root"] = Path("/content").resolve()

    ALLOWED_EXTENSIONS = {".pt", ".pth", ".db", ".sqlite", ".sqlite3"}

    @classmethod
    def validate_and_get_path(cls, requested_path: str) -> Path:
        """
        Validate path and return safe absolute path if it's within any allowed directory.
        """
        try:
            requested = Path(requested_path)
            full_path = requested.resolve()

            # Verify it's within at least one allowed directory
            is_allowed = False
            for base_dir in cls.ALLOWED_DIRS.values():
                if str(full_path).startswith(str(base_dir)):
                    is_allowed = True
                    break

            if not is_allowed:
                raise ValueError(f"Access denied or path traversal detected: {requested_path}")

            # Verify extension
            if full_path.suffix not in cls.ALLOWED_EXTENSIONS:
                raise ValueError(f"File type not allowed: {full_path.suffix}")

            return full_path

        except Exception as e:
            raise ValueError(f"Invalid path: {e}")


def position_analysis_tab(model, encoder, device):
    """Position analysis and visualization."""
    st.header("🔍 Position Analysis")
    
    fen_input = st.text_input("Enter FEN:", value=chess.Board().fen())
    
    if not fen_input:
        st.warning("Please enter a FEN string.")
        return

    try:
        board = chess.Board(fen_input)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            svg = render_board_svg(board, size=500)
            st.image(svg, use_container_width=True)
        
        with col2:
            if model and st.button("Analyze Position"):
                with st.spinner("Analyzing..."):
                    # Get model evaluation
                    tensor = encoder.board_to_tensor(board).unsqueeze(0).to(device)
                    with torch.no_grad():
                        policy_logits, value = model(tensor)
                    
                    st.metric("Position Evaluation", f"{value.item():.3f}")
                    
                    # Get top moves
                    mcts = MCTS(model, encoder, num_simulations=200)
                    best_move, stats = mcts.search(board, add_noise=False)
                    
                    st.subheader("Top Moves")
                    for move_data in stats.get('top_moves', [])[:5]:
                        st.write(f"**{move_data['move']}**: "
                                f"Visits={move_data['visits']}, "
                                f"Q={move_data['q_value']:.3f}")
    
    except Exception as e:
        st.error(f"Invalid FEN: {e}")


def main():
    """Main dashboard application."""
    setup_logging()

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", type=str, default="logs/training_logs.db", help="Path to SQLite metrics database")
    parser.add_argument("--checkpoint-path", type=str, default="checkpoints/latest.pt", help="Path to model checkpoint")

    # Handle streamlit-specific argument passing
    try:
        args, unknown = parser.parse_known_args()
    except Exception:
        # Fallback if argparse fails within streamlit
        class Args:
            db_path = "logs/training_logs.db"
            checkpoint_path = "checkpoints/latest.pt"
        args = Args()
    
    # Title
    st.title("♟️ Archimedes Chess AI Dashboard")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        db_path_raw = st.text_input("Database Path", value=args.db_path)
        checkpoint_path_raw = st.text_input("Checkpoint Path", value=args.checkpoint_path)
        
        try:
            db_path = str(SafePathValidator.validate_and_get_path(db_path_raw))
            checkpoint_path = str(SafePathValidator.validate_and_get_path(checkpoint_path_raw))
            checkpoint_path = resolve_checkpoint_path(checkpoint_path)

            # Show file status
            if not Path(db_path).exists():
                st.warning(f"⚠️ Database not found at: {db_path_raw}")
            if not Path(checkpoint_path).exists():
                st.warning(f"⚠️ Checkpoint not found at: {checkpoint_path_raw}")

        except ValueError as e:
            st.error(f"Security Error: {e}")
            st.stop()

        auto_refresh = st.checkbox("Auto Refresh", value=False)
        refresh_interval = 10
        if auto_refresh:
            refresh_interval = st.slider("Refresh Interval (s)", 5, 60, 10)
        
        st.markdown("---")
        st.header("📊 Quick Stats")
        
        # Load data
        data_loader = DashboardData(db_path)
        
        try:
            train_df = data_loader.get_training_metrics(limit=1)
            if not train_df.empty:
                latest = train_df.iloc[0]
                st.metric("Current Epoch", int(latest['epoch']))
                st.metric("Latest Loss", f"{latest['loss_total']:.4f}")
                st.metric("Top-1 Accuracy", f"{latest.get('accuracy_top1', 0)*100:.1f}%")
        except Exception:
            st.info("No training data yet")
    
    # Load model
    model, encoder, device = load_model(checkpoint_path)
    
    # Main tabs
    tab_titles = [
        "📈 Training", "🎯 MCTS", "♟️ Chess Performance",
        "💻 Hardware", "🎮 Play vs AI", "🔍 Analysis", "📥 Downloads"
    ]
    tabs = st.tabs(tab_titles)
    
    with tabs[0]: render_training_tab(data_loader)
    with tabs[1]: render_mcts_tab(data_loader)
    with tabs[2]: render_chess_tab(data_loader)
    with tabs[3]: render_hardware_tab(data_loader)
    with tabs[4]: play_vs_ai_tab(model, encoder, device)
    with tabs[5]: position_analysis_tab(model, encoder, device)
    with tabs[6]: render_downloads_tab(db_path)

    # Auto refresh
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()


def render_training_tab(data_loader):
    st.header("Training Metrics")
    try:
        train_df = data_loader.get_training_metrics(limit=1000)
        if not train_df.empty:
            train_df = train_df.sort_values('epoch')
            plot_training_loss(train_df)
            plot_accuracy(train_df)
            st.subheader("Recent Metrics")
            cols = ['epoch', 'loss_total', 'loss_policy', 'loss_value', 'accuracy_top1', 'learning_rate']
            display_df = train_df[cols].head(10)
            st.dataframe(display_df, use_container_width=True)
    except Exception as e:
        st.error(f"Error loading training data: {e}")


def render_mcts_tab(data_loader):
    st.header("MCTS Performance")
    try:
        mcts_df = data_loader.get_mcts_metrics(limit=1000)
        if not mcts_df.empty:
            mcts_df = mcts_df.sort_values('epoch')
            plot_mcts_performance(mcts_df)
    except Exception as e:
        st.error(f"Error loading MCTS data: {e}")


def render_chess_tab(data_loader):
    st.header("Chess Performance")
    try:
        chess_df = data_loader.get_chess_metrics(limit=1000)
        if not chess_df.empty:
            chess_df = chess_df.sort_values('epoch')
            plot_chess_performance(chess_df)
    except Exception as e:
        st.error(f"Error loading chess data: {e}")


def render_hardware_tab(data_loader):
    st.header("Hardware Utilization")
    try:
        hw_df = data_loader.get_hardware_metrics(limit=1000)
        if not hw_df.empty:
            hw_df = hw_df.sort_values('epoch')
            plot_hardware_usage(hw_df)
    except Exception as e:
        st.error(f"Error loading hardware data: {e}")


def render_downloads_tab(db_path):
    st.header("📥 Downloads")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Model Checkpoints")
        checkpoint_dir = Path("checkpoints")
        if checkpoint_dir.exists():
            checkpoints = list(checkpoint_dir.glob("*.pt"))
            for cp in checkpoints:
                try:
                    with open(cp, "rb") as f:
                        st.download_button(
                            label=f"📦 {cp.name}",
                            data=f.read(),
                            file_name=cp.name,
                            mime="application/octet-stream"
                        )
                except Exception as e:
                    st.error(f"Could not read {cp.name}: {e}")
    with col2:
        st.subheader("Game Records")
        if st.button("Export Games to PGN"):
            try:
                logger = MetricsLogger(db_path)
                logger.export_games_pgn("exported_games.pgn", limit=100)
                st.success("Games exported to exported_games.pgn")
                with open("exported_games.pgn", "rb") as f:
                    st.download_button(
                        label="Download PGN",
                        data=f.read(),
                        file_name="exported_games.pgn",
                        mime="text/plain"
                    )
            except Exception as e:
                st.error(f"Export failed: {e}")


if __name__ == "__main__":
    main()
