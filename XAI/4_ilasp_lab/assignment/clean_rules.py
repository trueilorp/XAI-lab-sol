import sys

# rule :- dist(X) < 5, guess(Y) > 3  --> this is ok (2 atoms, 2 comparisons)
# rule :- dist(X), guess(Y) < 5      --> this is not ok (2 atoms, 1 comparison)
def clean_rules(input_file, output_file):
	try:
		with open(input_file, 'r') as f:
			lines = f.readlines()
	except FileNotFoundError:
		print(f"Error: Could not find file '{input_file}'")
		return

	pruned_lines = []
	kept_count = 0
	removed_count = 0

	for line in lines:
		# Skip empty lines or comments
		if not line.strip() or line.strip().startswith('%'):
			continue

		# If line doesn't have a body (no ':-'), doesn't need filtering
		if ':-' not in line:
			pruned_lines.append(line)
			continue

		# Split into head and body
		parts = line.split(':-', 1)
		body = parts[1]

		# Count appearances of specific atoms
		# We use 'dist(' and 'guess(' to ensure we match the predicate, not just the word
		num_dist = body.count('dist(')
		num_guess = body.count('guess(')
		total_atoms = num_dist + num_guess

		# Count appearances of comparison operators
		num_less = body.count('<')
		num_greater = body.count('>')
		total_comparisons = num_less + num_greater
																				
		# The Condition: comparisons must equal atoms
		if total_atoms == total_comparisons:
			pruned_lines.append(line)
			kept_count += 1
		else:
			removed_count += 1

	with open(output_file, 'w') as f:
		f.writelines(pruned_lines)

	print(f"Finished processing.")
	print(f"Original rules: {len(lines)}")
	print(f"Rules kept:     {kept_count}")
	print(f"Rules removed:  {removed_count}")
	print(f"Cleaned rules written to: {output_file}")

if __name__ == "__main__":
	# You can change the filename here if needed
	clean_rules('s_m.txt', 's_m_pruned.txt')