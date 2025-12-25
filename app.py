import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go # We need this for the pie charts
import bcrypt
from datetime import date, timedelta, datetime
import time
import calendar # New import

# --- CONFIGURATION ---
st.set_page_config(page_title="AI Engineer Growth Path", page_icon="⚡", layout="wide")

# --- DATABASE MANAGER ---
def get_db():
    return st.connection("gsheets", type=GSheetsConnection)

def get_data(worksheet):
    try:
        df = get_db().read(worksheet=worksheet, ttl=0)
        return df.dropna(how="all")
    except:
        return pd.DataFrame()

def update_data(worksheet, df):
    get_db().update(worksheet=worksheet, data=df)

# --- ANALYTICS LOGIC ---
def calculate_streaks(df, user):
    """Calculates current login streak."""
    if df.empty: return 0
    
    user_logs = df[df['username'] == user].sort_values('date', ascending=False)
    if user_logs.empty: return 0
    
    # Get unique dates logged
    unique_dates = pd.to_datetime(user_logs['date']).dt.date.unique()
    
    today = date.today()
    if today not in unique_dates:
        # If haven't logged today yet, check if logged yesterday to keep streak alive
        if (today - timedelta(days=1)) not in unique_dates:
            return 0
    
    streak = 0
    check_date = today
    
    # Simple algorithm: check backwards day by day
    # (Adjust logic if you want to allow today to be skipped without breaking streak yet)
    if today in unique_dates:
        streak = 1
        check_date = today - timedelta(days=1)
    elif (today - timedelta(days=1)) in unique_dates:
        streak = 1
        check_date = today - timedelta(days=2)
        
    while check_date in unique_dates:
        streak += 1
        check_date -= timedelta(days=1)
        
    return streak


def plot_github_grid(df, user, year):
    """
    Generates a GitHub-style contribution heatmap for a specific year.
    - Colors: 0=Gray, 1=Light Green ... Max=Neon Green
    - X-Axis: Month names positioned correctly
    - Y-Axis: Mon/Wed/Fri labels
    """
    # 1. Filter Data for the specific year
    user_data = df[df['username'] == user].copy()
    user_data['date'] = pd.to_datetime(user_data['date'])
    user_data = user_data[user_data['date'].dt.year == year]
    
    # Count daily contributions
    daily_counts = user_data.groupby('date').size()
    
    # 2. Setup Date Range for the Full Year
    start_date = date(year, 1, 1)
    end_date = date(year, 12, 31)
    
    # Helper to find week number (0 to 52) and day of week (0=Mon to 6=Sun)
    def get_week_day(d):
        days_from_start = (d - start_date).days
        first_day_weekday = start_date.weekday()
        week = (days_from_start + first_day_weekday) // 7
        weekday = d.weekday()
        return week, weekday

    # 3. Build the Grid (7 Rows x 54 Cols)
    z_data = [[0]*54 for _ in range(7)]
    hover_text = [["" for _ in range(54)] for _ in range(7)]
    
    current = start_date
    while current <= end_date:
        wk, dy = get_week_day(current)
        if wk < 54:
            count = daily_counts.get(pd.Timestamp(current), 0)
            z_data[dy][wk] = count
            
            # Hover text
            date_str = current.strftime("%b %d, %Y")
            hover_text[dy][wk] = f"{count} contributions on {date_str}"
        current += timedelta(days=1)
        
    # 4. Define Colorscale
    max_val = max(daily_counts.max(), 1) if not daily_counts.empty else 1
    
    colors = [
        [0.0, "#161b22"],   # 0 (Empty) - Dark Gray
        [0.00001, "#0e4429"], # 1 (Lightest)
        [0.25, "#006d32"],
        [0.50, "#26a641"],
        [0.75, "#39d353"],
        [1.0, "#39d353"]    # Max (Brightest)
    ]

    # 5. Month Labels
    month_labels = []
    month_ticks = []
    for m in range(1, 13):
        d = date(year, m, 1)
        wk, _ = get_week_day(d)
        month_labels.append(d.strftime("%b"))
        month_ticks.append(wk)

    # 6. Create Plot
    fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=[i for i in range(54)],
        y=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
        text=hover_text,
        hoverinfo="text",
        colorscale=colors,
        showscale=False,
        xgap=2,  # Horizontal gap
        ygap=2,  # Vertical gap (The FIX is here)
        zmin=0,
        zmax=max_val
    ))
    
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=160,
        margin=dict(t=20, l=30, r=20, b=20),
        xaxis=dict(
            tickmode="array",
            tickvals=month_ticks,
            ticktext=month_labels,
            side="top",
            showgrid=False,
            zeroline=False,
            tickfont=dict(color="#8b949e", size=12)
        ),
        yaxis=dict(
            tickmode="array",
            tickvals=[1, 3, 5],
            ticktext=["Mon", "Wed", "Fri"],
            showgrid=False,
            zeroline=False,
            autorange="reversed",
            tickfont=dict(color="#8b949e", size=10)
        )
    )
    
    return fig

def plot_completion_pies(df, user, total_active_challenges):
    """
    Generates 3 Donut charts for Daily, Monthly, and Yearly completion.
    """
    user_df = df[df['username'] == user].copy()
    user_df['date'] = pd.to_datetime(user_df['date'])
    today = datetime.today()
    
    # --- CALCULATIONS ---
    
    # 1. Daily
    # Note: This is an estimation. "Daily Completion" usually means % of Today's tasks done.
    today_records = user_df[user_df['date'].dt.date == today.date()]
    daily_done = len(today_records)
    daily_total = total_active_challenges if total_active_challenges > 0 else 1 # Avoid div/0
    daily_val = (daily_done / daily_total) * 100
    
    # 2. Monthly
    this_month = user_df[user_df['date'].dt.month == today.month]
    monthly_done = len(this_month)
    # Estimate: Total possible = (Active Challenges * Days passed in month)
    days_in_month = today.day
    monthly_total = total_active_challenges * days_in_month
    monthly_val = (monthly_done / monthly_total * 100) if monthly_total > 0 else 0
    
    # 3. Yearly
    this_year = user_df[user_df['date'].dt.year == today.year]
    yearly_done = len(this_year)
    # Estimate: Total possible = (Active Challenges * Days passed in year)
    day_of_year = today.timetuple().tm_yday
    yearly_total = total_active_challenges * day_of_year
    yearly_val = (yearly_done / yearly_total * 100) if yearly_total > 0 else 0

    # --- PLOTTING ---
    
    fig = go.Figure()
    
    # Helper to create donut
    def create_donut(val, label, domain_x):
        # Cap value at 100 for visual sanity
        display_val = min(val, 100)
        remainder = 100 - display_val
        
        return go.Pie(
            values=[display_val, remainder],
            labels=["Done", "Remaining"],
            domain={'x': domain_x},
            hole=0.7,
            sort=False,
            marker=dict(colors=['#39d353', '#161b22']), # Green vs Dark Gray
            textinfo='none',
            hoverinfo='label+percent',
            name=label
        )

    fig.add_trace(create_donut(daily_val, "Daily", [0, 0.30]))
    fig.add_trace(create_donut(monthly_val, "Monthly", [0.35, 0.65]))
    fig.add_trace(create_donut(yearly_val, "Yearly", [0.70, 1.0]))

    # Add annotations in center
    fig.update_layout(
        annotations=[
            dict(text=f"{int(daily_val)}%", x=0.15, y=0.5, font_size=20, showarrow=False),
            dict(text=f"{int(monthly_val)}%", x=0.50, y=0.5, font_size=20, showarrow=False),
            dict(text=f"{int(yearly_val)}%", x=0.85, y=0.5, font_size=20, showarrow=False),
            dict(text="Daily", x=0.15, y=0.2, showarrow=False),
            dict(text="Monthly", x=0.50, y=0.2, showarrow=False),
            dict(text="Yearly", x=0.85, y=0.2, showarrow=False),
        ],
        showlegend=False,
        height=250,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# --- AUTHENTICATION ---
def verify_login(username, password):
    users = get_data("Users")
    if users.empty: return False
    user_row = users[users['username'] == username]
    if not user_row.empty:
        stored_hash = user_row.iloc[0]['password']
        if bcrypt.checkpw(password.encode('utf-8'), stored_hash.encode('utf-8')):
            return True
    return False

def register_user(username, password):
    users = get_data("Users")
    if not users.empty and username in users['username'].values:
        return False
    hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    new_user = pd.DataFrame([{"username": username, "password": hashed}])
    update_data("Users", pd.concat([users, new_user], ignore_index=True))
    return True

# --- CORE FEATURES ---
def get_user_challenges(username):
    df = get_data("User_Habits")
    if df.empty: return []
    # Only return active challenges (active == 1)
    return df[(df['username'] == username) & (df['active'] == 1)]['habit_name'].tolist()

def toggle_challenge(username, challenge_name, action="add"):
    df = get_data("User_Habits")
    
    if action == "add":
        # Check duplicates
        if not df.empty and not df[(df['username'] == username) & (df['habit_name'] == challenge_name)].empty:
            # If exists but inactive, reactivate it
            idx = df[(df['username'] == username) & (df['habit_name'] == challenge_name)].index
            df.at[idx[0], 'active'] = 1
        else:
            new_row = pd.DataFrame([{"username": username, "habit_name": challenge_name, "active": 1}])
            df = pd.concat([df, new_row], ignore_index=True)
            
    elif action == "remove":
        # We don't delete rows (to keep history), we just set active = 0
        if not df.empty:
            mask = (df['username'] == username) & (df['habit_name'] == challenge_name)
            df.loc[mask, 'active'] = 0
            
    update_data("User_Habits", df)

def log_progress(username, active_challenges, status_dict):
    df = get_data("Habits")
    today = date.today().isoformat()
    new_rows = []
    
    # Check if we already logged today to prevent duplicates (basic check)
    # Ideally, we would update existing rows, but append is safer for now
    
    for challenge in active_challenges:
        if status_dict.get(challenge):
            new_rows.append({
                "username": username,
                "date": today,
                "habit_name": challenge,
                "status": 1
            })
            
    if new_rows:
        update_data("Habits", pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True))
        return True
    return False

# --- UI PAGES ---
def login_page():
    st.markdown("<h1 style='text-align: center;'>⚡ AI Engineer Growth Path</h1>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1,2,1])
    
    with c2:
        tab1, tab2 = st.tabs(["Login", "Join the Quest"])
        with tab1:
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            if st.button("Enter Dashboard", type="primary"):
                if verify_login(u, p):
                    st.session_state['user'] = u
                    st.rerun()
                else:
                    st.error("Access Denied.")
        with tab2:
            nu = st.text_input("Pick a Username")
            np = st.text_input("Set Password", type="password")
            if st.button("Start Journey"):
                if register_user(nu, np):
                    st.success("Profile initialized. Login to begin.")
                else:
                    st.error("Username taken.")

def main_app():
    user = st.session_state['user']
    
    # --- SIDEBAR INFO ---
    with st.sidebar:
        st.title(f"👨‍💻 {user}")
        
        # Calculate Quick Stats
        history_df = get_data("Habits")
        streak = calculate_streaks(history_df, user)
        st.metric("🔥 Current Streak", f"{streak} Days")
        
        st.divider()
        nav = st.radio("Navigation", ["🚀 Daily Commit", "📈 Engineering Analytics", "⚙️ Config Challenges"])
        
        if st.button("Logout", use_container_width=True):
            del st.session_state['user']
            st.rerun()

    # --- TAB 1: DAILY COMMIT ---
    if nav == "🚀 Daily Commit":
        st.header(f"📝 Commit Log: {date.today()}")
        st.caption("Consistency is the key to mastering AI.")
        
        # 1. Get all active challenges
        all_challenges = get_user_challenges(user)
        
        # 2. Get what is ALREADY done today
        history_df = get_data("Habits")
        today_iso = date.today().isoformat()
        
        if not history_df.empty:
            # Filter for User + Today
            done_today = history_df[
                (history_df['username'] == user) & 
                (history_df['date'] == today_iso)
            ]['habit_name'].tolist()
        else:
            done_today = []

        # 3. Calculate Pending Challenges
        pending_challenges = [c for c in all_challenges if c not in done_today]
        
        # --- UI LOGIC ---
        if not all_challenges:
            st.warning("No active challenges found! Go to 'Config Challenges' to set your goals.")
            
        elif not pending_challenges:
            # If everything is done, show success state
            st.balloons()
            st.success("🎉 All daily commits pushed! Great work today.")
            
            with st.expander("View Today's Completed Commits"):
                for task in done_today:
                    st.write(f"✅ {task}")
                    
        else:
            # Show progress bar
            progress = len(done_today) / len(all_challenges)
            st.progress(progress, text=f"{len(done_today)}/{len(all_challenges)} Completed")
            
            with st.form("daily_log"):
                st.write("**Pending Tasks:**")
                status_dict = {}
                cols = st.columns(2)
                
                for i, challenge in enumerate(pending_challenges):
                    with cols[i % 2]:
                        st.markdown(f"**{challenge}**")
                        status_dict[challenge] = st.checkbox(f"Done", key=challenge)
                
                st.divider()
                submit = st.form_submit_button("Push Updates 🟢", use_container_width=True)
                
                if submit:
                    # Filter only checked boxes
                    to_log = [k for k, v in status_dict.items() if v]
                    
                    if to_log:
                        # Call log_progress with the list of habits to save
                        if log_progress(user, to_log, status_dict): 
                            st.success("Updates pushed!")
                            time.sleep(1)
                            st.rerun()
                    else:
                        st.warning("Select a task first.")

# --- TAB 2: ANALYTICS ---
    elif nav == "📈 Engineering Analytics":
        st.header("📊 Engineering Stats")
        
        history_df = get_data("Habits")
        
        if history_df.empty:
            st.info("No data yet. Go push some commits!")
        else:
            # --- YEAR SELECTOR (Like GitHub) ---
            # Get all unique years from data + Current Year
            history_df['date'] = pd.to_datetime(history_df['date'])
            available_years = sorted(history_df['date'].dt.year.unique(), reverse=True)
            current_year = date.today().year
            if current_year not in available_years:
                available_years.insert(0, current_year)
            
            # Layout: Graph on Left, Year Buttons on Right
            col_graph, col_years = st.columns([5, 1])
            
            with col_years:
                st.write("### Year")
                selected_year = st.radio("Select Year", available_years, label_visibility="collapsed")
            
            with col_graph:
                st.subheader(f"{selected_year} Contributions")
                
                # CALL THE NEW FUNCTION
                fig_grid = plot_github_grid(history_df, user, selected_year)
                if fig_grid:
                    st.plotly_chart(fig_grid, use_container_width=True, config={'displayModeBar': False})
            
            st.divider()
            
            # 2. Completion Pie Charts
            st.subheader("Completion Rates")
            active_challenges_count = len(get_user_challenges(user))
            fig_pies = plot_completion_pies(history_df, user, active_challenges_count)
            st.plotly_chart(fig_pies, use_container_width=True)



    # --- TAB 3: CONFIG CHALLENGES ---
    elif nav == "⚙️ Config Challenges":
        st.header("🛠️ Configure Your Stack")
        st.caption("Add the skills or habits you want to track daily.")
        
        c1, c2 = st.columns([3, 1])
        with c1:
            new_chal = st.text_input("New Challenge Name", placeholder="e.g., Read 1 AI Paper, Leetcode Medium")
        with c2:
            st.write("")
            st.write("")
            if st.button("Add Challenge"):
                if new_chal:
                    toggle_challenge(user, new_chal, "add")
                    st.success(f"Added {new_chal}")
                    time.sleep(0.5)
                    st.rerun()
        
        st.divider()
        st.subheader("Active Challenges")
        current = get_user_challenges(user)
        for chal in current:
            c1, c2 = st.columns([4, 1])
            c1.markdown(f"🔹 **{chal}**")
            if c2.button("Archive", key=f"del_{chal}"):
                toggle_challenge(user, chal, "remove")
                st.rerun()

# --- APP FLOW ---
if 'user' not in st.session_state:
    login_page()
else:
    main_app()