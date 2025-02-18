import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
st.image("https://storage.googleapis.com/mle-courses-prod/users/61b6fa1ba83a7e37c8309756/private-files/678dadd0-603b-11ef-b0a7-998b84b38d43-ProtonX_logo_horizontally__1_.png", width=100)  # Replace "logo.png" with your logo file path


# Function to build the 2D rotation matrix
def build_rotation_matrix(theta):
    """Returns the 2D rotation matrix for angle theta."""
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])

# Function to generate LaTeX for rotation matrix
def generate_rotation_matrix_latex(theta):
    """Generates LaTeX representation of the 2D rotation matrix with theta value."""
    latex_matrix = rf"""
    R(\theta) =
    \begin{{bmatrix}}
    \cos({theta:.2f}) & -\sin({theta:.2f}) \\
    \sin({theta:.2f}) & \cos({theta:.2f})
    \end{{bmatrix}}
    """
    return latex_matrix


# Function to apply Rotary Position Encoding (ROPE)
def apply_rope(embedding, position, base_theta=0.45):
    """Applies ROPE to the embedding based on position."""
    rotated_embedding = embedding.copy()
    theta = position * base_theta
    for i in range(0, len(embedding) - 1, 2):  # Process in pairs
        R = build_rotation_matrix(theta)
        rotated_embedding[i:i+2] = R @ embedding[i:i+2]
    return rotated_embedding

# Function to calculate the angle between vectors
def calculate_angle_between_vectors(v1, v2):
    """Calculates the angle between two vectors."""
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    return np.arccos(dot_product / (norm_v1 * norm_v2))

# Function to visualize embeddings with selectable m
def visualize_embeddings(word, embedding, rotated_embedding, m):
    """Visualizes original and rotated embeddings with arrows and arcs that properly connect vectors."""
    fig, ax = plt.subplots(figsize=(6,6))

    first_orgi = np.array(embedding[:2])
    second_orgi = np.array(embedding[2:])
    k_0_1_rotated = np.array(rotated_embedding[:2])
    k_2_3_rotated = np.array(rotated_embedding[2:])

    # Function to compute angle and draw arc correctly connecting vectors
    def draw_arc(vector1, vector2, arc_radius=0.3, color='green'):
        """Draws an arc between two vectors correctly positioned at the origin."""
        # Compute angle
        dot_product = np.dot(vector1, vector2)
        norm_v1 = np.linalg.norm(vector1)
        norm_v2 = np.linalg.norm(vector2)
        angle_rad = np.arccos(dot_product / (norm_v1 * norm_v2))
        angle_deg = np.degrees(angle_rad)

        # Determine start and end angles for the arc
        start_angle = np.arctan2(vector1[1], vector1[0])
        end_angle = np.arctan2(vector2[1], vector2[0])

        if end_angle < start_angle:
            end_angle += 2 * np.pi

        arc_theta = np.linspace(start_angle, end_angle, 30)
        arc_x = arc_radius * np.cos(arc_theta)
        arc_y = arc_radius * np.sin(arc_theta)
        ax.plot(arc_x, arc_y, color=color, linestyle='-', linewidth=1.5)

        # Annotate angle
        mid_angle = (start_angle + end_angle) / 2
        mid_x = arc_radius * np.cos(mid_angle)
        mid_y = arc_radius * np.sin(mid_angle)
        ax.text(mid_x, mid_y, f'{angle_deg:.2f}°', fontsize=10, color=color, ha='center')

    # Plot original vectors
    ax.quiver(0, 0, first_orgi[0], first_orgi[1], angles='xy', scale_units='xy', scale=1, color='blue', label='Original Vector 0-1')
    ax.text(first_orgi[0], first_orgi[1], f'[{first_orgi[0]:.2f}, {first_orgi[1]:.2f}]', fontsize=7, ha='left')

    ax.quiver(0, 0, second_orgi[0], second_orgi[1], angles='xy', scale_units='xy', scale=1, color='blue', label='Original Vector 2-3')
    ax.text(second_orgi[0], second_orgi[1], f'[{second_orgi[0]:.2f}, {second_orgi[1]:.2f}]', fontsize=7, ha='left')

    # Plot rotated vectors
    ax.quiver(0, 0, k_0_1_rotated[0], k_0_1_rotated[1], angles='xy', scale_units='xy', scale=1, color='red', label='Rotated Vector 0-1')
    ax.text(k_0_1_rotated[0], k_0_1_rotated[1], f'[{k_0_1_rotated[0]:.2f}, {k_0_1_rotated[1]:.2f}]', fontsize=7, ha='left')

    ax.quiver(0, 0, k_2_3_rotated[0], k_2_3_rotated[1], angles='xy', scale_units='xy', scale=1, color='red', label='Rotated Vector 2-3')
    ax.text(k_2_3_rotated[0], k_2_3_rotated[1], f'[{k_2_3_rotated[0]:.2f}, {k_2_3_rotated[1]:.2f}]', fontsize=7, ha='left')

    # Draw arcs for both rotations, ensuring they connect vectors
    draw_arc(first_orgi, k_0_1_rotated, arc_radius=0.3, color='green')
    draw_arc(second_orgi, k_2_3_rotated, arc_radius=0.3, color='purple')

    # Set plot limits
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_title(f"Vector Rotation Visualization for '{word}' (m={m})")
    ax.legend()
    ax.grid(True)
    ax.set_aspect('equal')
    
    ax.legend(fontsize=3)
    st.pyplot(fig)

# Streamlit UI
st.markdown("### Rotary position embedding visualization")


# Sample sentence and embeddings
sentence = ["the", "cat", "sat", "on", "the", "mat"]
embeddings = np.array([
    [0.1, 0.2, -0.3, -0.4],
    [0.5, 0.6, -0.5, 0.5],  # Reference row
    [0.3, 0.2, -0.3, 0.2],
    [0.4, 0.3, -0.2, 0.8],
    [0.2, 0.5, -0.1, 0.3],
    [0.7, 0.1, -0.3, 0.4]
])

# Use session state to store selected word position
if "selected_position" not in st.session_state:
    st.session_state.selected_position = 0  # Default to first word

# Render buttons for each word
cols = st.columns(len(sentence))
for i, word in enumerate(sentence):
    if cols[i].button(word, key=f"word_button_{i}"):  # Unique key for each button
        st.session_state.selected_position = i  # Store selected position in session state

# Get the selected word and its embedding
selected_position = st.session_state.selected_position
selected_word = sentence[selected_position]
original_embedding = embeddings[selected_position]
rotated_embedding = apply_rope(original_embedding, selected_position)



# Compute theta for the selected position
theta_value = selected_position * 1.2  # Base theta from the ROPE function

# Display LaTeX matrix in Streamlit
st.markdown("Rotation Matrix")

st.latex(generate_rotation_matrix_latex(theta_value))

# Visualize embeddings
visualize_embeddings(selected_word, original_embedding, rotated_embedding, selected_position)


# Show embeddings before and after ROPE
col1, col2 = st.columns(2)
with col1:
    st.write("**Original Embedding**")
    st.write(original_embedding)
with col2:
    st.write("**After ROPE Rotation**")
    st.write(rotated_embedding)
st.markdown("[Paper](https://arxiv.org/pdf/2104.09864v5)")