Extract resume understanding, job-description understanding, and skill-gap analysis from the texts below.

Return a JSON object with exactly these top-level keys:
- "resume": candidate name, summary, skills, technologies, projects, experience, education, certifications, achievements, years_of_experience
- "job_description": role_title, company, summary, required_skills, preferred_skills, responsibilities, technologies, tools, keywords, experience_level
- "skill_gap_analysis": matching_skills, missing_or_unclear_skills, strong_signals, interview_focus_areas

Resume:
{resume_text}

Job Description:
{jd_text}

