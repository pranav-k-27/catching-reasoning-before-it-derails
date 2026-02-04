import matplotlib.pyplot as plt

tasks = ["Business", "Logic", "Ethics", "Paradox"]
student_only = [500, 500, 500, 500]
supervised = [150, 150, 150, 150]

plt.figure()
plt.bar(tasks, student_only)
plt.bar(tasks, supervised)
plt.ylabel("Approximate Tokens Generated")
plt.title("Inference-Time Token Usage")
plt.legend(["Student-only", "Process-Supervised"])
plt.tight_layout()
plt.show()
