# Quick Python script to fix indentation in pipeline.py
import sys

file_path = r'd:\github\USCore\src\ucorefs\processing\pipeline.py'

with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Lines are 0-indexed, so line 428 in editor is index 427
# Fix lines that have 12 spaces when they should have 20 spaces (inside if/else blocks)
fixes = {
    427: lines[427].replace('            old_state_name', '                    old_state_name'),
    428: lines[428].replace('            logger.info', '                    logger.info'),
    434: lines[434].replace('            state_name', '                    state_name'),
    435: lines[435].replace('            logger.warning', '                    logger.warning'),
}

for idx, fixed_line in fixes.items():
    lines[idx] = fixed_line

with open(file_path, 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("Indentation fixed successfully!")
