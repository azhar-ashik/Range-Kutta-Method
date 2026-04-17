import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Custom CSS for UI styling
st.set_page_config(page_title="RK4 Solver", layout="wide")
st.title("🔢 Runge-Kutta 4th Order ODE Solver")
st.markdown("Solve first-order ODEs of the form: $dy/dx = f(x, y)$")

# Sidebar for User Inputs
st.sidebar.header("User Inputs")
equation = st.sidebar.text_input("Enter f(x, y):", "x + y")
x0 = st.sidebar.number_input("Initial x (x0):", value=0.0)
y0 = st.sidebar.number_input("Initial y (y0):", value=1.0)
xn = st.sidebar.number_input("Final x (Target):", value=2.0)
h = st.sidebar.number_input("Step size (h):", value=0.1, format="%.4f")

def solve_rk4(f_str, x0, y0, xn, h):
    # Safe evaluation environment
    allowed_names = {"x": 0, "y": 0, "np": np, "sin": np.sin, "cos": np.cos, "exp": np.exp, "log": np.log, "pow": pow}
    
    x_vals = [x0]
    y_vals = [y0]
    steps = int((xn - x0) / h)
    
    x, y = x0, y0
    
    for _ in range(steps):
        # Calculate k values
        k1 = eval(f_str, {"__builtins__": None}, {**allowed_names, "x": x, "y": y})
        k2 = eval(f_str, {"__builtins__": None}, {**allowed_names, "x": x + 0.5*h, "y": y + 0.5*h*k1})
        k3 = eval(f_str, {"__builtins__": None}, {**allowed_names, "x": x + 0.5*h, "y": y + 0.5*h*k2})
        k4 = eval(f_str, {"__builtins__": None}, {**allowed_names, "x": x + h, "y": y + h*k3})
        
        y = y + (h/6.0) * (k1 + 2*k2 + 2*k3 + k4)
        x = x + h
        
        x_vals.append(x)
        y_vals.append(y)
        
    return x_vals, y_vals

if st.sidebar.button("Solve"):
    try:
        x_pts, y_pts = solve_rk4(equation, x0, y0, xn, h)
        
        # Display Results in columns
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Calculation Table")
            df = pd.DataFrame({"x": x_pts, "y": y_pts})
            st.dataframe(df, use_container_width=True)
            
        with col2:
            st.subheader("Solution Curve")
            fig, ax = plt.subplots()
            ax.plot(x_pts, y_pts, marker='o', linestyle='-', color='b', label=f"y' = {equation}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
            
    except Exception as e:
        st.error(f"Error in expression: {e}. Use Python syntax (e.g., x*y or np.sin(x))")
