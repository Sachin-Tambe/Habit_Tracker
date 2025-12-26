import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import bcrypt
from datetime import date, timedelta, datetime
import time
import calendar

# --- CONFIGURATION ---
st.set_page_config(page_title="AI Growth Engine", page_icon="⚡", layout="wide")

# --- 1. GAMIFICATION ENGINE ---
LEVELS = {
    1: "Hatchling 🐣", 5: "Busy Beaver 🦫", 10: "Swift Gazelle 🦌",
    20: "Loyal Wolf 🐺", 40: "Roaring Lion 🦁", 60: "Wise Owl 🦉",
    80: "Titan Elephant 🐘", 100: "Mythic Dragon 🐉"
}

def calculate_streaks(df, user):
    if df.empty: return 0
    user_logs = df[df['username'] == user].sort_values('date', ascending=False)
    if user_logs.empty: return 0
    unique_dates = pd.to_datetime(user_logs['date']).dt.date.unique()
    today = date.today()
    streak = 0
    check_date = today
    if today in unique_dates:
        streak += 1
        check_date -= timedelta(days=1)
    elif (today - timedelta(days=1)) in unique_dates:
        check_date -= timedelta(days=1)
    while check_date in unique_dates:
        streak += 1
        check_date -= timedelta(days=1)
    return streak

def calculate_gamification(df, user):
    if df.empty: return 1, LEVELS[1], 0, 0
    user_logs = df[df['username'] == user]
    total_xp = len(user_logs) * 10
    current_level = int(total_xp / 100) + 1
    current_title = LEVELS[1]
    sorted_levels = sorted(LEVELS.keys(), reverse=True)
    for lvl in sorted_levels:
        if current_level >= lvl:
            current_title = LEVELS[lvl]
            break
    progress = min((total_xp % 100) / 100, 1.0)
    return current_level, current_title, progress, total_xp

def get_badges(df, user):
    badges = []
    if df.empty: return badges
    user_df = df[df['username'] == user].copy()
    user_df['date'] = pd.to_datetime(user_df['date'])
    total_logs = len(user_df)
    
    if total_logs >= 1: badges.append("🌱 First Step")
    if total_logs >= 50: badges.append("🔨 Builder")
    if total_logs >= 100: badges.append("🛡️ Centurion")
    
    streak = calculate_streaks(df, user)
    if streak >= 7: badges.append("🔥 Week Wielder")
    if streak >= 30: badges.append("🌕 Monthly Master")
    
    all_habit_names = " ".join(user_df['habit_name'].unique()).lower()
    if "code" in all_habit_names or "python" in all_habit_names: badges.append("💻 Hacker")
    if "git" in all_habit_names: badges.append("🐙 Git Master")
    
    return badges

# --- 2. JOURNEY LOGIC (THE 90 DAY QUEST) ---
def get_journey_plan():
    try:
        df = get_db().read(worksheet="90 Days Journey", ttl=600) 
        return df.dropna(how="all")
    except:
        return pd.DataFrame()

def get_journey_history():
    expected_cols = ["username", "Day", "Phase 1 (The Gym)", "Phase 2 (The Lab)", "Phase 3 (The Show)"]
    try:
        df = get_db().read(worksheet="90 Days Journey Completed", ttl=5) # Cached
        if df.empty or "Day" not in df.columns:
            return pd.DataFrame(columns=expected_cols)
        df['Day'] = pd.to_numeric(df['Day'], errors='coerce').fillna(0).astype(int)
        return df
    except Exception:
        return pd.DataFrame(columns=expected_cols)

def get_active_journey_tasks(user, journey_history_df, journey_plan_df):
    if journey_plan_df.empty: return 0, []
    
    if journey_history_df.empty:
        user_history = pd.DataFrame(columns=["username", "Day", "Phase 1 (The Gym)", "Phase 2 (The Lab)", "Phase 3 (The Show)"])
    else:
        user_history = journey_history_df[journey_history_df['username'] == user]

    for index, row in journey_plan_df.iterrows():
        try:
            day_num = int(row.iloc[0])
        except:
            continue
        
        t1, t2, t3 = row.iloc[1], row.iloc[2], row.iloc[3]
        tasks_map = {t1: "Phase 1 (The Gym)", t2: "Phase 2 (The Lab)", t3: "Phase 3 (The Show)"}
        
        day_record = user_history[user_history['Day'] == day_num]
        pending = []
        
        if day_record.empty:
            pending = [t for t in [t1, t2, t3] if pd.notna(t) and t != ""]
        else:
            record = day_record.iloc[0]
            for task, col_name in tasks_map.items():
                if pd.notna(task) and task != "":
                    val = record.get(col_name)
                    if pd.isna(val) or val == "" or val is None:
                        pending.append(task)
        
        if pending:
            return day_num, pending
            
    return 91, []

# --- 3. VISUALIZATION ENGINE (FIXED) ---

def plot_github_grid(df, user, year):
    """
    Fixed GitHub Grid with proper Month labels on X-axis and Mon/Wed/Fri on Y-axis.
    """
    user_data = df[df['username'] == user].copy()
    user_data['date'] = pd.to_datetime(user_data['date'])
    user_data = user_data[user_data['date'].dt.year == year]
    daily_counts = user_data.groupby('date').size()
    
    # Generate full year range
    start_date = date(year, 1, 1)
    end_date = date(year, 12, 31)
    
    # Helper to calculate (Week, Day) coordinates
    def get_week_day(d):
        days_from_start = (d - start_date).days
        # Find first day of the year's weekday to shift logic
        # 0=Mon, 6=Sun
        first_day_weekday = start_date.weekday()
        
        # Calculate week number (0-53)
        week = (days_from_start + first_day_weekday) // 7
        weekday = d.weekday()
        return week, weekday

    # 54 Weeks (Columns), 7 Days (Rows)
    z_data = [[0]*54 for _ in range(7)]
    hover_text = [["" for _ in range(54)] for _ in range(7)]
    
    current = start_date
    while current <= end_date:
        wk, dy = get_week_day(current)
        if wk < 54:
            count = daily_counts.get(pd.Timestamp(current), 0)
            z_data[dy][wk] = count
            date_str = current.strftime("%b %d, %Y")
            hover_text[dy][wk] = f"{count} contributions on {date_str}"
        current += timedelta(days=1)
        
    max_val = max(daily_counts.max(), 1) if not daily_counts.empty else 1
    
    # GitHub Green Scale
    colors = [
        [0.0, "#161b22"],   # Empty (Dark Gray)
        [0.0001, "#0e4429"], # 1 commit
        [0.25, "#006d32"],
        [0.50, "#26a641"],
        [0.75, "#39d353"],
        [1.0, "#39d353"]    # Brightest
    ]
    
    # X-Axis Month Labels
    month_labels = []
    month_ticks = []
    for m in range(1, 13):
        d = date(year, m, 1)
        wk, _ = get_week_day(d)
        month_labels.append(d.strftime("%b"))
        month_ticks.append(wk)

    fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=[i for i in range(54)],
        y=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
        text=hover_text,
        hoverinfo="text",
        colorscale=colors,
        showscale=False,
        xgap=2, # Gap between squares
        ygap=2,
        zmin=0,
        zmax=max_val
    ))
    
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=180,
        margin=dict(t=30, l=40, r=20, b=20), # Extra left margin for labels
        xaxis=dict(
            tickmode="array",
            tickvals=month_ticks,
            ticktext=month_labels,
            side="top", # Months on TOP
            showgrid=False,
            zeroline=False,
            tickfont=dict(color="#8b949e", size=12)
        ),
        yaxis=dict(
            tickmode="array",
            tickvals=[1, 3, 5], # Mon, Wed, Fri indices
            ticktext=["Mon", "Wed", "Fri"],
            showgrid=False,
            zeroline=False,
            autorange="reversed", # Mon on top
            tickfont=dict(color="#8b949e", size=11)
        )
    )
    return fig

def plot_habit_breakdown_bars(df, user, year, month=None):
    """
    Cleaner Bar Chart with readable labels.
    """
    user_df = df[df['username'] == user].copy()
    user_df['date'] = pd.to_datetime(user_df['date'])
    user_df = user_df[user_df['date'].dt.year == year]
    
    title = f"📅 Yearly Breakdown ({year})"
    if month:
        user_df = user_df[user_df['date'].dt.month == month]
        month_name = calendar.month_name[month]
        title = f"📅 Monthly Breakdown ({month_name} {year})"
    
    if user_df.empty: return None
    
    habit_counts = user_df['habit_name'].value_counts().reset_index()
    habit_counts.columns = ['Habit', 'Commits']
    
    # Truncate very long habit names for display
    habit_counts['ShortName'] = habit_counts['Habit'].apply(lambda x: x[:30] + "..." if len(x) > 30 else x)
    
    fig = px.bar(
        habit_counts, 
        x='ShortName', # Use short name for X axis
        y='Commits',
        title=title, 
        text_auto=True, 
        color='Habit', # Different colors
        hover_data=['Habit'] # Show full name on hover
    )
    
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", 
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False, # Hide legend if it's too messy
        xaxis_title="",
        xaxis=dict(tickangle=-45) # Angle text so it doesn't overlap
    )
    return fig

def plot_correlation_matrix(df, user):
    user_df = df[df['username'] == user].copy()
    if user_df.empty: return None

    # Pivot the data
    pivot = user_df.pivot_table(index='date', columns='habit_name', values='status', aggfunc='max').fillna(0)
    
    # Check if we have enough data
    if len(pivot.columns) < 2: return None
    
    # Calculate correlation
    corr = pivot.corr()
    
    # TRUNCATE LABELS: Shorten long names for the chart only
    # Example: "Sleep Well - (7+ Hrs)..." -> "Sleep Well..."
    short_labels = [col[:15] + "..." if len(col) > 15 else col for col in corr.columns]
    
    # Create Heatmap
    fig = px.imshow(
        corr, 
        text_auto=".2f", # Show 2 decimal places
        aspect="equal",  # FORCE SQUARE SQUARES
        color_continuous_scale="RdBu_r", 
        title="🔮 Habit Correlation Matrix",
        x=short_labels, # Use short names on X axis
        y=short_labels  # Use short names on Y axis
    )
    
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", 
        plot_bgcolor="rgba(0,0,0,0)",
        height=400, # Fixed height to prevent squishing
        margin=dict(l=10, r=10, t=50, b=10)
    )
    
    # Fix the hover data so you can still see the full name when hovering
    fig.update_traces(
        hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>Correlation: %{z:.2f}<extra></extra>"
    )
    
    return fig


def plot_journey_analytics(journey_plan_df, journey_history_df, user):
    if journey_plan_df.empty: return None
    
    col_map = {
        "Phase 1 (The Gym)": journey_plan_df.columns[1],
        "Phase 2 (The Lab)": journey_plan_df.columns[2],
        "Phase 3 (The Show)": journey_plan_df.columns[3]
    }
    
    user_history = pd.DataFrame()
    if not journey_history_df.empty:
        user_history = journey_history_df[journey_history_df['username'] == user]
    
    stats = []
    
    for phase_name, plan_col in col_map.items():
        total = journey_plan_df[plan_col].count()
        done = 0
        if not user_history.empty and phase_name in user_history.columns:
            done = user_history[phase_name].count()
            
        short_name = phase_name.split(":")[0] + " (" + phase_name.split("(")[-1].replace(")", "")
        
        stats.append({"Phase": short_name, "Type": "Completed", "Count": done})
        stats.append({"Phase": short_name, "Type": "Remaining", "Count": max(0, total - done)})

    df_stats = pd.DataFrame(stats)
    
    fig = px.bar(
        df_stats, x="Count", y="Phase", color="Type", orientation='h',
        title="🛡️ Quest Completion by Phase",
        text_auto=True,
        color_discrete_map={"Completed": "#39d353", "Remaining": "#161b22"}
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False), yaxis=dict(showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

# --- DATABASE & AUTH (WITH CACHING TO FIX 429 ERROR) ---
def get_db(): return st.connection("gsheets", type=GSheetsConnection)

@st.cache_data(ttl=5) # CACHING ADDED HERE
def get_data(worksheet):
    try: return get_db().read(worksheet=worksheet, ttl=5).dropna(how="all")
    except: return pd.DataFrame()

def update_data(worksheet, df): 
    get_db().update(worksheet=worksheet, data=df)
    get_data.clear() # CLEAR CACHE ON WRITE

def verify_login(username, password):
    users = get_data("Users")
    if users.empty: return False
    user_row = users[users['username'] == username]
    if not user_row.empty:
        return bcrypt.checkpw(password.encode('utf-8'), user_row.iloc[0]['password'].encode('utf-8'))
    return False

def register_user(username, password):
    users = get_data("Users")
    if not users.empty and username in users['username'].values: return False
    hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    update_data("Users", pd.concat([users, pd.DataFrame([{"username": username, "password": hashed}])], ignore_index=True))
    return True

def get_user_challenges(username):
    df = get_data("User_Habits")
    if df.empty: return []
    return df[(df['username'] == username) & (df['active'] == 1)]['habit_name'].tolist()

def toggle_challenge(username, challenge_name, action="add"):
    df = get_data("User_Habits")
    if action == "add":
        if not df.empty and not df[(df['username'] == username) & (df['habit_name'] == challenge_name)].empty:
            idx = df[(df['username'] == username) & (df['habit_name'] == challenge_name)].index
            df.at[idx[0], 'active'] = 1
        else:
            df = pd.concat([df, pd.DataFrame([{"username": username, "habit_name": challenge_name, "active": 1}])], ignore_index=True)
    elif action == "remove":
        if not df.empty:
            mask = (df['username'] == username) & (df['habit_name'] == challenge_name)
            df.loc[mask, 'active'] = 0
    update_data("User_Habits", df)

def log_progress(user, challenges_to_log):
    df = get_data("Habits")
    today = date.today().isoformat()
    new_rows = []
    for challenge in challenges_to_log:
        new_rows.append({"username": user, "date": today, "habit_name": challenge, "status": 1})
    if new_rows:
        update_data("Habits", pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True))
        return True
    return False

def log_journey_progress(user, day_num, tasks_dict, journey_plan_df):
    df = get_journey_history()
    
    day_plan = journey_plan_df[journey_plan_df.iloc[:, 0] == day_num]
    if day_plan.empty: return False
    
    t1, t2, t3 = day_plan.iloc[0, 1], day_plan.iloc[0, 2], day_plan.iloc[0, 3]
    col_gym, col_lab, col_show = "Phase 1 (The Gym)", "Phase 2 (The Lab)", "Phase 3 (The Show)"
    
    mask = (df['username'] == user) & (df['Day'] == day_num)
    new_data = {}
    if t1 in tasks_dict: new_data[col_gym] = t1
    if t2 in tasks_dict: new_data[col_lab] = t2
    if t3 in tasks_dict: new_data[col_show] = t3
    
    if df[mask].empty:
        new_row = {"username": user, "Day": day_num}
        new_row.update(new_data)
        updated_df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        idx = df[mask].index[0]
        for col, val in new_data.items():
            df.at[idx, col] = val
        updated_df = df
        
    update_data("90 Days Journey Completed", updated_df)
    return True

# --- UI: MAIN APP ---
def main_dashboard():
    user = st.session_state['user']
    history_df = get_data("Habits")
    
    with st.sidebar:
        level, title, progress, total_xp = calculate_gamification(history_df, user)
        badges = get_badges(history_df, user)
        st.title(f"👨‍💻 {user}")
        st.write(f"**{title}** (Lvl {level})")
        st.progress(progress, text=f"XP: {total_xp}")
        with st.expander(f"🏆 Trophy Case ({len(badges)})", expanded=True):
            if badges: st.write("\n".join([f"- {b}" for b in badges]))
            else: st.caption("No badges yet.")
        st.divider()
        nav = st.radio("Navigation", ["🚀 Daily Commit", "📊 My Profile & Analytics", "⚙️ Config Challenges"])
        if st.button("Logout"):
            del st.session_state['user']
            st.rerun()

    if nav == "🚀 Daily Commit":
        st.header(f"📝 Commit Log: {date.today()}")
        
        # --- 90 DAYS JOURNEY LOGIC ---
        if user == "sachin":
            journey_plan_df = get_journey_plan()
            journey_hist_df = get_journey_history() 
            
            if not journey_plan_df.empty:
                current_day, pending_tasks = get_active_journey_tasks(user, journey_hist_df, journey_plan_df)
                
                with st.expander("🗺️ Journey Map & Stats", expanded=False):
                    col_map, col_stats = st.columns([1, 2])
                    with col_map:
                        fig_gauge = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = current_day - 1,
                            title = {'text': "Days Conquered"},
                            gauge = {'axis': {'range': [None, 90]}, 'bar': {'color': "#39d353"}}
                        ))
                        fig_gauge.update_layout(height=200, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor="rgba(0,0,0,0)")
                        st.plotly_chart(fig_gauge, use_container_width=True)
                        
                    with col_stats:
                        fig_phase = plot_journey_analytics(journey_plan_df, journey_hist_df, user)
                        if fig_phase: st.plotly_chart(fig_phase, use_container_width=True)

                if current_day <= 90:
                    st.info(f"📍 **Day {current_day} Objective**")
                    with st.form("journey_form"):
                        st.write("Complete these to unlock the next level:")
                        j_status = {}
                        cols = st.columns(3)
                        for i, t in enumerate(pending_tasks):
                            j_status[t] = cols[i].checkbox(f"{t}")
                        
                        if st.form_submit_button("Push Journey Update 🚀", use_container_width=True):
                            to_log = {k: v for k,v in j_status.items() if v}
                            if to_log and log_journey_progress(user, current_day, to_log, journey_plan_df):
                                st.balloons()
                                st.success("Journey Updated!")
                                time.sleep(1)
                                st.rerun()
                    st.divider()
                else:
                    st.success("🎉 YOU HAVE COMPLETED THE 90 DAY JOURNEY!")
                    st.divider()

        # --- STANDARD HABITS LOGIC ---
        all_challenges = get_user_challenges(user)
        today_iso = date.today().isoformat()
        
        # 1. Get ALL records for today
        raw_done_today = []
        if not history_df.empty:
            raw_done_today = history_df[(history_df['username'] == user) & (history_df['date'] == today_iso)]['habit_name'].tolist()
        
        # 2. Filter: Only count habits that are ACTIVE
        valid_done_today = [h for h in raw_done_today if h in all_challenges]
        
        # 3. Pending: Active habits NOT in the done list
        pending = [c for c in all_challenges if c not in raw_done_today]
        
        if not all_challenges and user != "sachin":
             st.warning("No challenges active! Go to Config to add some.")
        elif not pending:
            st.success("✅ Daily Habits Completed!")
            with st.expander("View Completed Today"):
                for t in valid_done_today: st.write(f"✅ {t}")
        else:
            st.subheader("Your Routine")
            total_active = len(all_challenges)
            completed_active = len(valid_done_today)
            progress_val = min(completed_active / max(total_active, 1), 1.0)
            
            st.progress(progress_val, text=f"{completed_active}/{total_active} Done")
            
            with st.form("log"):
                status = {}
                cols = st.columns(2)
                for i, c in enumerate(pending):
                    status[c] = cols[i%2].checkbox(c)
                if st.form_submit_button("Push Habit Updates 🟢", use_container_width=True):
                    to_log = [k for k,v in status.items() if v]
                    if to_log and log_progress(user, to_log):
                        st.success(f"Updates pushed! +{len(to_log)*10} XP")
                        time.sleep(1)
                        st.rerun()

    elif nav == "📊 My Profile & Analytics":
        st.header("🧠 My Growth Dashboard")
        if history_df.empty:
            st.info("No data available yet.")
        else:
            history_df['date'] = pd.to_datetime(history_df['date'])
            years = sorted(history_df['date'].dt.year.unique(), reverse=True)
            current_year = date.today().year
            if current_year not in years: years.insert(0, current_year)
            
            c1, c2 = st.columns([5, 1])
            with c2:
                st.write("")
                sel_year = st.selectbox("Year", years, index=years.index(current_year))
            with c1:
                st.subheader(f"Contribution Map ({sel_year})")
                grid_fig = plot_github_grid(history_df, user, sel_year)
                if grid_fig: st.plotly_chart(grid_fig, use_container_width=True, config={'displayModeBar': False})
            
            st.divider()
            st.subheader("📊 Habit Breakdown")
            col_month, col_year = st.columns(2)
            current_month = date.today().month
            
            with col_month:
                bar_fig_month = plot_habit_breakdown_bars(history_df, user, sel_year, current_month)
                if bar_fig_month: st.plotly_chart(bar_fig_month, use_container_width=True)
                else: st.info(f"No data for {calendar.month_name[current_month]} {sel_year}.")
            
            with col_year:
                bar_fig_year = plot_habit_breakdown_bars(history_df, user, sel_year)
                if bar_fig_year: st.plotly_chart(bar_fig_year, use_container_width=True)
                else: st.info(f"No data for {sel_year}.")
                
            st.divider()
            st.subheader("🔮 Correlation Lab")
            corr_fig = plot_correlation_matrix(history_df, user)
            if corr_fig: st.plotly_chart(corr_fig, use_container_width=True)
            else: st.info("Need more habit variety to show correlations.")

    elif nav == "⚙️ Config Challenges":
        st.header("🛠️ Configure Stack")
        c1, c2 = st.columns([3, 1])
        with c1: new_chal = st.text_input("New Challenge Name")
        with c2: 
            st.write("")
            st.write("")
            if st.button("Add"):
                if new_chal:
                    toggle_challenge(user, new_chal, "add")
                    st.success(f"Added {new_chal}")
                    time.sleep(0.5)
                    st.rerun()
        st.divider()
        st.subheader("Active Challenges")
        for chal in get_user_challenges(user):
            c1, c2 = st.columns([4, 1])
            c1.markdown(f"🔹 **{chal}**")
            if c2.button("Archive", key=f"del_{chal}"):
                toggle_challenge(user, chal, "remove")
                st.rerun()

if __name__ == "__main__":
    if 'user' not in st.session_state:
        st.title("⚡ AI Growth Engine")
        tab1, tab2 = st.tabs(["Login", "Register"])
        with tab1:
            u = st.text_input("User")
            p = st.text_input("Pass", type="password")
            if st.button("Login"):
                if verify_login(u, p):
                    st.session_state['user'] = u
                    st.rerun()
                else: st.error("Invalid credentials")
        with tab2:
            nu = st.text_input("New User")
            np = st.text_input("New Pass", type="password")
            if st.button("Register"):
                if register_user(nu, np): st.success("Created! Login now.")
                else: st.error("Username taken")
    else:
        main_dashboard()
