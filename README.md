# Rotary Position Embedding Visualization

![](https://storage.googleapis.com/mle-courses-prod/users/61b6fa1ba83a7e37c8309756/private-files/a642f280-edba-11ef-9acf-6f7079495a92-Screen_Shot_2025_02_18_at_12.38.51.png)

This project is a **Streamlit** app that visualizes **Rotary Position Embeddings (ROPE)**, a technique used in Transformers to encode positional information more effectively.

## Features
- **Rotary Position Encoding (ROPE)** implementation with visual representation.
- **Interactive word selection** to apply rotation to embeddings.
- **LaTeX representation** of the rotation matrix.
- **Vector visualization** showing how ROPE rotates embeddings in 2D space.

## Installation

Ensure you have Python 3.7+ installed. Then, install dependencies:
```bash
pip install streamlit numpy matplotlib
```

## Usage
Run the Streamlit app with:
```bash
streamlit run app.py
```

## How It Works
### 1. **Rotation Matrix Calculation**
The 2D rotation matrix is computed as:
\[
R(\theta) = \begin{bmatrix} \cos(\theta) & -\sin(\theta) \\ \sin(\theta) & \cos(\theta) \end{bmatrix}
\]
This matrix is applied to the embedding vectors.

### 2. **Applying ROPE to Embeddings**
Given an embedding and its position, the rotation matrix is used to rotate the embedding in 2D space.

### 3. **Visualization**
The app:
- Displays the rotation matrix using LaTeX.
- Visualizes original and rotated vectors.
- Draws arcs to indicate the rotation angle.

## UI Interaction
- Click on a word to apply ROPE to its embedding.
- Observe the original and rotated embeddings.
- View how the embedding is transformed via the rotation matrix.

## Reference
- Original ROPE Paper: [Rotary Position Embedding](https://arxiv.org/pdf/2104.09864v5)

## License
MIT License