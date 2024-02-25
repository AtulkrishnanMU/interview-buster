import streamlit as st
import os
import cv2
import sqlite3

conn = sqlite3.connect('users.db')
c = conn.cursor()

c.execute('''
            DELETE FROM users
            WHERE username != ?
            ''', ('admin',))  # Keep the admin entry (if any) and remove others
conn.commit()
print("Existing entries removed successfully!")