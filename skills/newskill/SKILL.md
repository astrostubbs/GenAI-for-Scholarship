---
name: newskill
description: Interactively create a new Claude Code skill through conversation. Gathers requirements, then iteratively drafts and refines the skill with user feedback until it's right.
argument-hint: [skill-name]
disable-model-invocation: true
allowed-tools: Read, Write, Bash, Glob, AskUserQuestion
---

# Create a New Skill

You are helping the user build a new Claude Code skill. The skill name provided is: **$ARGUMENTS**

If no skill name was provided, ask the user for one first.

## Step 1: Quick Configuration

Use AskUserQuestion to gather these structural choices (combine into 1-2 questions):

1. **Scope**: Personal (all projects, ~/.claude/skills/) or Project (this repo only, .claude/skills/)?
2. **Invocation**: User only, Claude only, or both?
3. **Execution**: Inline or forked subagent? If forked, which agent type?
4. **Model**: Inherit, opus, sonnet, or haiku?
5. **Arguments**: What arguments does it take? (e.g., [filename], [issue-number])

## Step 2: Describe What the Skill Should Do

Now ask the user to describe in their own words — in as much detail as they want — what this skill should do when invoked. Tell them:

"Describe what this skill should do. Be as detailed as you like — what steps should it take, what output should it produce, what format should it follow, any specific instructions or constraints. I'll draft the skill from your description and we'll refine it together."

Wait for the user's full description. Do NOT use AskUserQuestion for this — let them type freely in the conversation.

## Step 3: Draft the SKILL.md

Based on everything gathered, produce a complete first draft of the SKILL.md file. Show it to the user in full — both the YAML frontmatter and the markdown body.

After showing the draft, ask: "What would you change? I can adjust the structure, add or remove sections, change the tone, make instructions more or less specific — anything. Or say 'looks good' to save it."

## Step 4: Iterative Refinement

This is the critical step. Repeat this loop:

1. The user gives feedback (add this section, remove that, rephrase this, make it more specific, etc.)
2. You revise the SKILL.md accordingly
3. Show the updated version in full
4. Ask again: "Any more changes, or ready to save?"

Continue until the user says it's ready. Do NOT rush to save — the user should feel they can keep refining as long as they want.

During refinement, actively help:
- If instructions are vague, suggest ways to make them more specific
- If the scope is too broad, suggest breaking it into multiple skills
- If something would work better as a different frontmatter option, say so
- Point out potential issues (e.g., "this needs WebFetch in allowed-tools if you want it to search the web")

## Step 5: Save

Once the user approves:

1. Create the directory at the correct location (personal or project scope)
2. Write the SKILL.md file
3. Create any supporting files discussed during refinement
4. Confirm: "Saved to [path]. You can invoke it with /[name] in your next session (or restart this one)."
