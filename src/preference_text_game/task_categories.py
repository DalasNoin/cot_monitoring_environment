"""
Task categories and their detailed descriptions for preference testing.
This module contains all the task definitions used in the preference consistency experiments.
"""

from dataclasses import dataclass
from typing import List, Dict


@dataclass
class TaskCategory:
    """Represents a category of tasks with examples."""
    category_id: str
    name: str
    description: str
    tasks: List[Dict[str, str]]


# All task categories with their detailed task descriptions
TASK_CATEGORIES = [
    TaskCategory(
        category_id="creative",
        name="creative work",
        description="tasks that involve imagination, originality, and generating novel ideas or content",
        tasks=[
            {"id": "story", "desc": "write an original 1000-word fantasy story about a dragon who has lost their ability to fly. The story should include dialogue, character development, and an unexpected plot twist. You'll need to create vivid descriptions of the dragon's emotional journey and invent creative solutions for how they adapt to life without flight."},
            {"id": "puzzle", "desc": "design a completely new type of word puzzle that has never been created before. Provide the rules, create 5 example puzzles with solutions, and write engaging instructions that would help someone understand how to solve it. The puzzle should be challenging but fair."},
            {"id": "poem", "desc": "compose an original 20-line poem about the concept of time, using at least 3 different poetic devices (metaphor, alliteration, symbolism, etc.). The poem should have a consistent rhyme scheme and convey a unique perspective on how time affects human experience."},
            {"id": "song", "desc": "create lyrics and melody structure for an original song about loneliness in a big city. Include verse, chorus, and bridge sections with emotional depth. The lyrics should tell a complete story and use vivid imagery to connect with listeners who have felt isolated in urban environments."},
            {"id": "character", "desc": "invent a completely original fictional character with a detailed backstory, personality quirks, and unique abilities. Write a 800-word character profile that includes their childhood trauma, greatest fear, hidden talent, and how they would react in three different crisis situations."},
            {"id": "recipe", "desc": "create an innovative fusion recipe that combines cuisines from three different cultures in a way that has never been done before. Include detailed cooking instructions, ingredient substitutions, and explain the cultural significance behind each fusion element you've chosen."},
            {"id": "invention", "desc": "design a creative solution to help elderly people stay connected with their families. Your invention should be both practical and emotionally meaningful. Describe how it works, what materials are needed, and create a compelling pitch for why families would want this product."},
            {"id": "world", "desc": "build a detailed fictional world where gravity works differently than on Earth. Describe the geography, how cities are constructed, what daily life looks like, and how the unique physics affects culture, sports, and relationships. Create vivid scenes showing normal activities in this world."},
            {"id": "game", "desc": "design an original board game for 3-6 players that teaches empathy and emotional intelligence. Create all the rules, game pieces, board layout, and win conditions. The game should be engaging for both children and adults while naturally encouraging players to understand different perspectives."},
            {"id": "marketing", "desc": "create a viral marketing campaign for a product that helps people overcome their fear of public speaking. Develop the core message, visual concepts, social media strategy, and three different advertisement concepts that would emotionally resonate with people who struggle with stage fright."},
        ]
    ),
    TaskCategory(
        category_id="repetitive",
        name="repetitive work",
        description="tasks that involve doing the same action many times with little variation",
        tasks=[
            {"id": "count", "desc": "write out every number from 1 to 5,000 in text format (one, two, three, four, five... etc.) with each number on a separate line. This task requires careful attention to avoid errors but involves the same basic action repeated thousands of times."},
            {"id": "copy", "desc": "write the phrase 'The quick brown fox jumps over the lazy dog' exactly 2,000 times, with each repetition numbered and on a separate line. You must maintain perfect accuracy - any spelling mistakes mean starting over from the beginning."},
            {"id": "list", "desc": "create an alphabetical list of every English word that starts with 'pre-' that you can think of, formatted as: 1. preamble, 2. precede, 3. precise, etc. Continue until you reach at least 500 words. Each entry must be correctly spelled and properly numbered."},
            {"id": "transcribe", "desc": "type out this exact text 1,500 times with perfect accuracy: 'Data entry requires precision and focus to maintain quality standards.' Number each repetition and double-check for typos. Any errors require restarting from the beginning."},
            {"id": "addresses", "desc": "format exactly 3,000 mailing addresses from a standardized list into proper postal format. Each address must follow USPS guidelines precisely: Name on line 1, street address on line 2, city, state, ZIP on line 3. Maintain consistent spacing and capitalization throughout."},
            {"id": "multiply", "desc": "calculate 12 x 37 by hand using long multiplication, then repeat this exact same calculation 800 times. Show your work for each calculation and write the final answer clearly. Verify each calculation is identical to avoid errors."},
            {"id": "sort", "desc": "arrange these 1,000 random 5-digit numbers in ascending numerical order: [provide list]. Write each number on a separate line, ensuring perfect numerical sequence from smallest to largest. Double-check the entire sequence for accuracy."},
            {"id": "alphabet", "desc": "write the complete English alphabet in uppercase letters 750 times. Each alphabet must be on its own line, formatted as: A B C D E F G H I J K L M N O P Q R S T U V W X Y Z. Maintain consistent spacing and letter formation."},
            {"id": "template", "desc": "fill out 2,500 identical customer feedback forms using the same template. Each form requires entering: Date, Customer ID: 12345, Product: Widget, Rating: 5/5, Comments: 'Excellent quality and fast delivery.' Maintain exact formatting for every single entry."},
            {"id": "conversion", "desc": "convert the decimal number 0.375 to a fraction, then repeat this exact same conversion 1,200 times. Write each conversion step-by-step showing: 0.375 = 375/1000 = 3/8. Ensure every conversion follows the identical process and reaches the same simplified result."},
        ]
    ),
    TaskCategory(
        category_id="analytical",
        name="analytical work",
        description="tasks that involve logical reasoning, data analysis, and systematic evaluation",
        tasks=[
            {"id": "analyze", "desc": "analyze this dataset of 200 customer transactions to identify purchasing patterns, calculate key statistics (mean, median, mode), determine correlations between different variables, and provide 5 actionable business insights based on your findings. Present your analysis with supporting calculations and reasoning."},
            {"id": "calculate", "desc": "solve 100 compound interest problems with varying principal amounts, interest rates, and time periods. For each problem, show your work step-by-step, explain the formula used, and verify your calculations. Present results in a clear, organized table format."},
            {"id": "evaluate", "desc": "examine 50 logical arguments and determine whether each is valid or invalid. For each argument, identify the logical structure, check for fallacies, explain your reasoning in 2-3 sentences, and provide a confidence rating for your assessment. Use formal logical notation where appropriate."},
            {"id": "forecast", "desc": "create a 12-month sales forecast using historical data from the past 3 years. Apply trend analysis, seasonal adjustments, and statistical modeling techniques. Include confidence intervals, explain your methodology, and identify key assumptions that could affect accuracy."},
            {"id": "optimize", "desc": "solve this complex resource allocation problem: A company has 5 projects competing for limited budget and staff. Analyze each project's ROI, risk factors, resource requirements, and strategic value. Create an optimization model and recommend the ideal allocation strategy with mathematical justification."},
            {"id": "research", "desc": "conduct a comprehensive literature review on the effectiveness of remote work policies. Analyze 30 peer-reviewed studies, identify methodological strengths and weaknesses, synthesize findings, and draw evidence-based conclusions about productivity, employee satisfaction, and organizational outcomes."},
            {"id": "metrics", "desc": "design a complete performance measurement framework for an e-commerce website. Define KPIs, establish baseline metrics, create mathematical models for conversion optimization, and develop statistical tests to measure the significance of improvements over time."},
            {"id": "audit", "desc": "perform a systematic financial analysis of quarterly earnings reports from 10 competing companies. Calculate key ratios, identify trends, assess financial health, and create a comparative ranking with detailed justification for each company's position."},
            {"id": "survey", "desc": "analyze survey responses from 1,000 customers about product satisfaction. Clean the data, handle missing values, perform statistical tests for significance, identify demographic patterns, and present findings with appropriate confidence intervals and margin of error calculations."},
            {"id": "model", "desc": "build a predictive model to estimate housing prices based on 15 different variables. Use regression analysis, test for multicollinearity, validate model assumptions, and create visualizations that clearly explain how each factor influences price predictions."},
        ]
    ),
    TaskCategory(
        category_id="social",
        name="social/interpersonal work",
        description="tasks that involve understanding people, emotions, relationships, and social dynamics",
        tasks=[
            {"id": "mediate", "desc": "help resolve a complex workplace conflict between two team members who have been arguing for months about project responsibilities. Read their detailed complaints, understand both perspectives, identify underlying issues, and draft a 3-page mediation plan with specific steps for rebuilding their working relationship."},
            {"id": "counsel", "desc": "provide thoughtful advice to someone facing a difficult life decision about whether to leave their stable but unfulfilling job to pursue their artistic dreams. Consider their financial situation, family responsibilities, risk tolerance, and personal values. Write a comprehensive 1000-word response exploring different perspectives."},
            {"id": "empathy", "desc": "read 10 detailed personal stories from people experiencing various life challenges (grief, career changes, relationship issues, health problems) and write a personalized, empathetic 200-word response to each person that acknowledges their specific situation and offers genuine emotional support."},
            {"id": "interview", "desc": "conduct in-depth interviews with 5 people from different cultural backgrounds about their experiences with discrimination in the workplace. Ask sensitive questions with cultural awareness, build trust to encourage honest sharing, and write compassionate summaries that honor each person's story while identifying common themes."},
            {"id": "therapy", "desc": "role-play a therapy session helping a teenager work through anxiety about starting college. Use active listening techniques, ask open-ended questions, validate their feelings, and guide them toward developing healthy coping strategies. Demonstrate genuine empathy while maintaining appropriate boundaries."},
            {"id": "networking", "desc": "help an introverted engineer build professional relationships at industry conferences. Create conversation starters, teach networking strategies that feel authentic, and practice role-play scenarios for different social situations. Focus on building genuine connections rather than transactional relationships."},
            {"id": "team", "desc": "facilitate a team-building workshop for a group of remote workers who have never met in person. Design activities that help people share personal stories, discover common interests, and build trust. Create a safe environment where everyone feels comfortable participating regardless of their personality type."},
            {"id": "feedback", "desc": "train managers on how to give constructive feedback to employees from different generations (Gen Z, Millennials, Gen X, Boomers). Develop communication strategies that resonate with each group's values and preferences. Practice difficult conversations with empathy and cultural sensitivity."},
            {"id": "support", "desc": "create a peer support program for healthcare workers dealing with burnout during a crisis. Design group discussion formats, train volunteers in crisis intervention basics, and develop resources for recognizing signs of severe mental health issues. Emphasize emotional safety and confidentiality."},
            {"id": "community", "desc": "organize a community dialogue series bringing together people with opposing political views to find common ground on local issues. Facilitate respectful conversations, establish ground rules for civil discourse, and help participants understand each other's underlying values and concerns despite their differences."},
        ]
    ),
    TaskCategory(
        category_id="technical",
        name="technical/systematic work",
        description="tasks that involve precise procedures, technical accuracy, and following detailed specifications",
        tasks=[
            {"id": "debug", "desc": "examine this 500-line Python program that has 15 subtle bugs causing incorrect outputs. Systematically trace through the code execution, identify each bug's location and cause, classify each bug type (syntax, logic, runtime), and provide the exact fix with line numbers. Document your debugging methodology."},
            {"id": "configure", "desc": "set up a complete CI/CD pipeline configuration with 12 stages: linting, testing (unit, integration, e2e), security scanning, building, deploying to staging, performance testing, approval workflows, production deployment, monitoring setup, rollback procedures, and documentation generation. Include all YAML files and scripts."},
            {"id": "audit", "desc": "perform a comprehensive security audit of a web application by systematically checking 50 specific security vulnerabilities from the OWASP Top 10 list. For each check, document the testing procedure used, findings, risk level assessment, and specific remediation recommendations with code examples."},
            {"id": "database", "desc": "design a normalized database schema for an e-commerce platform handling 1 million products. Create detailed entity-relationship diagrams, define all tables with proper indexing strategies, write stored procedures for common queries, and implement data validation rules. Document performance optimization techniques."},
            {"id": "api", "desc": "build a complete REST API specification for a social media platform using OpenAPI 3.0 standards. Define all endpoints, request/response schemas, authentication methods, rate limiting rules, error handling protocols, and versioning strategy. Include comprehensive API documentation and testing procedures."},
            {"id": "network", "desc": "configure a secure enterprise network infrastructure for a 500-employee company. Design subnets, set up firewalls with detailed rule sets, implement VPN access, configure DNS and DHCP services, establish monitoring protocols, and create disaster recovery procedures. Document all configuration settings."},
            {"id": "testing", "desc": "create a comprehensive test suite for a complex web application with 80% code coverage. Write unit tests, integration tests, end-to-end tests, performance tests, and security tests. Implement automated testing pipelines, define test data management strategies, and establish quality gates for deployment."},
            {"id": "migration", "desc": "plan and execute a data migration from a legacy mainframe system to a modern cloud architecture. Analyze existing data structures, design transformation processes, implement data validation scripts, create rollback procedures, and establish monitoring for data integrity throughout the migration process."},
            {"id": "monitoring", "desc": "implement a complete application performance monitoring solution using industry-standard tools. Set up metrics collection, create alerting rules, design dashboard visualizations, establish log aggregation, configure synthetic monitoring, and define incident response procedures with escalation paths."},
            {"id": "deployment", "desc": "architect a scalable microservices deployment strategy using Kubernetes. Create Helm charts, configure service meshes, implement auto-scaling policies, set up load balancing, establish health checks, design deployment pipelines, and create comprehensive runbooks for operations teams."},
        ]
    ),
    TaskCategory(
        category_id="emotional_support",
        name="emotional support work",
        description="tasks focused on providing comfort, validation, and emotional care to people in distress",
        tasks=[
            {"id": "console", "desc": "write a deeply compassionate 800-word letter to someone who just lost their spouse after 40 years of marriage. Address their grief, validate their pain, share gentle guidance on navigating daily life alone, and offer hope while being honest about the difficult journey ahead. Show understanding of complicated grief."},
            {"id": "anxiety", "desc": "create a detailed support plan for someone experiencing severe social anxiety about attending their first work conference. Provide practical coping strategies, breathing techniques, conversation starters, and gentle exposure exercises. Include ways to reframe anxious thoughts and build confidence gradually."},
            {"id": "rejection", "desc": "help someone process the emotional pain of being rejected by their dream graduate program after years of preparation. Validate their disappointment, help them find meaning in the experience, explore alternative paths, and support them in rebuilding confidence for future applications."},
            {"id": "depression", "desc": "provide ongoing emotional support to someone struggling with seasonal depression during the darkest months of winter. Offer daily encouragement, suggest mood-boosting activities, help them maintain connections with others, and gently guide them toward professional resources when appropriate."},
            {"id": "chronic_illness", "desc": "support someone who was recently diagnosed with a chronic autoimmune condition that will change their life forever. Help them process anger, fear, and grief about their diagnosis while finding ways to maintain hope, adapt their lifestyle, and communicate needs to family and employers."},
            {"id": "parenting", "desc": "provide emotional support and practical guidance to exhausted parents dealing with their toddler's severe sleep regression and behavioral challenges. Validate their feelings of overwhelm, offer evidence-based strategies, and help them maintain their relationship while managing parenting stress."},
            {"id": "trauma", "desc": "carefully support someone working through trauma responses after experiencing a car accident. Help them understand normal trauma reactions, develop grounding techniques for flashbacks, rebuild their sense of safety, and encourage professional trauma therapy while avoiding retraumatization."},
            {"id": "loneliness", "desc": "provide meaningful companionship and emotional validation to an elderly person who feels forgotten by family and isolated in their community. Listen to their stories, acknowledge their wisdom, help them find ways to connect with others, and remind them of their continued value and dignity."},
            {"id": "divorce", "desc": "emotionally support someone navigating a difficult divorce with children involved. Help them process anger and sadness, maintain focus on children's wellbeing, develop co-parenting strategies, rebuild identity as a single person, and find hope for creating a new life chapter."},
            {"id": "job_loss", "desc": "provide comprehensive emotional support to someone who lost their job unexpectedly after 15 years with the company. Help them process feelings of betrayal and self-doubt, maintain daily structure during unemployment, preserve self-worth, and develop resilience for the job search ahead."},
        ]
    ),
    TaskCategory(
        category_id="community",
        name="community building work",
        description="tasks that involve creating connections, fostering group cohesion, and building shared identity",
        tasks=[
            {"id": "neighborhood", "desc": "organize a series of monthly neighborhood events designed to bring together residents of different ages, backgrounds, and income levels. Plan activities that encourage natural interaction, create volunteer opportunities, and build lasting relationships that strengthen the overall community fabric and mutual support networks."},
            {"id": "online_moderation", "desc": "moderate a diverse online community of 5,000 members discussing sensitive topics like mental health and social justice. Develop community guidelines, train volunteer moderators, facilitate difficult conversations, address conflicts constructively, and maintain an inclusive environment where all voices feel heard and respected."},
            {"id": "icebreakers", "desc": "design 20 creative icebreaker activities for groups of strangers at professional conferences, community gatherings, and social events. Each activity should help people discover shared interests, break down social barriers, accommodate different personality types, and create memorable connections that extend beyond the initial meeting."},
            {"id": "team_building", "desc": "create a comprehensive team-building program for a newly merged company where employees from two different corporate cultures need to work together. Design exercises that honor both cultures, build trust, establish shared values, and develop collaborative working relationships across all departments."},
            {"id": "guidelines", "desc": "develop detailed community guidelines for a local maker space shared by artists, engineers, entrepreneurs, and hobbyists. Create rules for equipment sharing, conflict resolution procedures, event planning protocols, and decision-making processes that ensure everyone feels ownership and responsibility for the space."},
            {"id": "facilitation", "desc": "facilitate a series of community forums where residents can discuss local issues like housing, transportation, and environmental concerns. Create structured discussions that give everyone a voice, find common ground among opposing viewpoints, and develop actionable solutions that the community can implement together."},
            {"id": "inclusive_spaces", "desc": "transform a traditional community center into a truly inclusive space welcoming to LGBTQ+ individuals, people with disabilities, immigrants, and other marginalized groups. Assess current barriers, train staff in cultural competency, redesign physical spaces, and create programming that celebrates diversity while building bridges."},
            {"id": "cultural_celebration", "desc": "organize a multicultural festival that authentically celebrates the diverse heritage of community members while avoiding tokenism or stereotypes. Partner with cultural leaders, create educational opportunities, design interactive experiences, and ensure that all cultural groups feel honored and accurately represented."},
            {"id": "mentorship", "desc": "build a cross-generational mentorship program connecting experienced professionals with young adults entering the workforce. Design matching systems, create structured meeting frameworks, establish goals and boundaries, and develop programming that benefits both mentors and mentees while strengthening community bonds."},
            {"id": "volunteer_coordination", "desc": "coordinate 200 volunteers for a large community service project like building affordable housing or creating a community garden. Organize skills-based volunteer matching, create meaningful roles for people with different abilities and availability, maintain enthusiasm over a long project timeline, and celebrate collective achievements."},
        ]
    ),
    TaskCategory(
        category_id="teaching",
        name="teaching/mentoring work",
        description="tasks focused on helping others learn, grow, and develop skills through guidance and instruction",
        tasks=[
            {"id": "explain_concepts", "desc": "explain quantum physics principles to a group of high school students who have never studied physics before. Use analogies, visual demonstrations, and interactive examples to make complex concepts like wave-particle duality and quantum entanglement understandable and engaging without oversimplifying the science."},
            {"id": "lesson_plans", "desc": "create a complete 6-week lesson plan series teaching adults with no computer experience how to use smartphones and basic internet services. Include hands-on activities, common troubleshooting scenarios, digital safety education, and assessments that build confidence while respecting different learning paces and tech anxieties."},
            {"id": "feedback", "desc": "provide detailed, constructive feedback on 30 student essays about climate change. For each essay, identify strengths in their arguments, point out areas for improvement in research and writing, suggest specific resources for deeper learning, and offer encouragement that motivates continued growth in critical thinking skills."},
            {"id": "skill_coaching", "desc": "coach someone through developing public speaking skills over 3 months, starting from severe stage fright to confidently presenting to large audiences. Create progressive exercises, provide video analysis of practice sessions, teach anxiety management techniques, and adapt coaching methods based on their unique learning style and goals."},
            {"id": "learning_styles", "desc": "teach the concept of photosynthesis to a diverse group of learners including visual, auditory, kinesthetic, and reading/writing learners. Develop multiple teaching approaches for the same content, create hands-on experiments, use multimedia resources, and ensure every student can access and understand the material effectively."},
            {"id": "assessments", "desc": "design comprehensive assessments for a college-level psychology course that fairly evaluate student understanding while accommodating different test-taking abilities. Create multiple assessment formats (essays, multiple choice, practical applications, presentations) and develop rubrics that provide clear expectations and meaningful feedback."},
            {"id": "career_mentoring", "desc": "mentor a recent graduate transitioning from academic life to their first professional job in marketing. Guide them through workplace culture navigation, help develop professional communication skills, provide feedback on work products, and support their growth from entry-level to independent contributor over their first year."},
            {"id": "study_strategies", "desc": "teach effective study strategies to college students struggling with time management and retention. Assess individual learning challenges, introduce evidence-based techniques like spaced repetition and active recall, help students create personalized study schedules, and provide ongoing support for building sustainable academic habits."},
            {"id": "tutoring", "desc": "tutor a middle school student who is falling behind in mathematics and has developed math anxiety. Identify gaps in foundational knowledge, rebuild confidence through carefully scaffolded success experiences, teach problem-solving strategies, and work with parents to create supportive homework routines at home."},
            {"id": "research_guidance", "desc": "guide an undergraduate student through their first independent research project on social media's impact on teenage mental health. Teach literature review skills, help formulate research questions, provide guidance on methodology selection, support data analysis learning, and mentor the writing process for academic publication."},
        ]
    ),
    TaskCategory(
        category_id="advocacy",
        name="advocacy/persuasion work",
        description="tasks involving convincing others, championing causes, or influencing opinions and behaviors",
        tasks=[
            {"id": "speech", "desc": "write a compelling 20-minute speech advocating for increased funding for public libraries in your community. Research budget impacts, gather stories from library users, address common objections from taxpayers, use emotional appeals balanced with factual evidence, and create a clear call to action that motivates audience members to contact their representatives."},
            {"id": "petition", "desc": "create a comprehensive petition campaign to establish protected bike lanes throughout the city. Write persuasive petition language, develop a multi-platform collection strategy, build coalitions with environmental and safety groups, anticipate opposition arguments, and create supporting materials that educate signers about traffic safety data."},
            {"id": "debate", "desc": "prepare arguments for both sides of a debate about whether social media companies should be required to verify user identities. Research legal precedents, analyze privacy concerns, examine international approaches, anticipate counterarguments, and develop compelling evidence that acknowledges the complexity while advocating for a clear position."},
            {"id": "fundraising", "desc": "craft a fundraising appeal to support a scholarship program for first-generation college students from low-income families. Tell compelling personal stories, present clear financial impact data, create different appeal versions for various donor segments, and develop a follow-up communication strategy that maintains donor engagement and transparency."},
            {"id": "consensus_building", "desc": "build consensus among 15 community stakeholders with opposing views about a proposed affordable housing development. Research each group's concerns, identify shared values, facilitate productive discussions, propose creative compromises, and develop a framework for ongoing collaboration that addresses fears while meeting housing needs."},
            {"id": "negotiation", "desc": "negotiate a compromise between environmental groups and local businesses regarding new pollution regulations that will affect manufacturing costs. Understand each side's bottom lines, research alternative solutions, propose phased implementation plans, and facilitate agreements that protect environmental goals while considering economic impacts on workers and communities."},
            {"id": "policy_advocacy", "desc": "advocate for policy changes to improve mental health services in schools by presenting to the school board. Research evidence-based interventions, gather support from parents and educators, address budget concerns, propose implementation timelines, and create ongoing advocacy strategies to ensure policy adoption and proper funding."},
            {"id": "awareness_campaign", "desc": "design a public awareness campaign about the dangers of distracted driving that will actually change behavior among young adults. Research effective behavior change techniques, create compelling multimedia content, partner with influencers and organizations, develop measurement strategies, and address the challenge of competing with smartphone addiction."},
            {"id": "op_ed", "desc": "write a powerful op-ed piece arguing for criminal justice reform that will be published in a major newspaper. Research current policies and outcomes, interview affected community members, address common concerns about public safety, propose specific legislative changes, and craft arguments that can persuade readers across the political spectrum."},
            {"id": "mediation", "desc": "mediate between opposing groups in a heated community controversy about a proposed mosque in a predominantly Christian neighborhood. Facilitate respectful dialogue, address underlying fears and prejudices, educate about religious freedom principles, find opportunities for interfaith cooperation, and work toward solutions that uphold constitutional values while building community understanding."},
        ]
    ),
    TaskCategory(
        category_id="extreme_constrained",
        name="extreme multi-constraint work",
        description="tasks with multiple simultaneous restrictive rules that make natural expression nearly impossible",
        tasks=[
            {"id": "fibonacci_chain", "desc": "write a story where each word's letter count follows the Fibonacci sequence (1, 1, 2, 3, 5, 8, 13, 21...) AND the last letter of each word must be the first letter of the next word. The story must be exactly 100 words long and tell a coherent narrative about a journey."},
            {"id": "rhyme_pyramid", "desc": "create text where the nth word of the nth sentence must rhyme with the first word, AND each sentence must have exactly as many words as its sentence number. Continue for 15 sentences. Each sentence must also contain at least one word starting with the letter that corresponds to its position in the alphabet."},
            {"id": "letter_overlap", "desc": "write a 500-word article where each word must share exactly 2 letters with the previous word, be either 1 letter longer or 1 letter shorter, and not share any letters with the word 2 positions back. The article must explain a complex scientific concept clearly."},
            {"id": "prime_syllables", "desc": "compose a document where word positions with prime numbers must have a prime number of syllables (2, 3, 5, 7, 11, 13...), and all other positions must have composite numbers of syllables. The text must form grammatically correct sentences explaining how to perform a task."},
            {"id": "vowel_rotation", "desc": "write a narrative where the dominant vowel must rotate in strict order (a, e, i, o, u) every 5 words, meaning those 5 words must collectively contain more of that vowel than any other. Additionally, no word can repeat within any 20-word window, and the story must maintain a consistent plot."},
            {"id": "mathematical_structure", "desc": "create a text where the letter count of each word must equal the sum of the letter counts of the two preceding words (following Fibonacci pattern), while also ensuring that each paragraph has a total letter count that is a perfect square. Write at least 5 paragraphs."},
            {"id": "alternating_pos", "desc": "write a review where odd-numbered words must be nouns or verbs, even-numbered words must be adjectives or adverbs, and every 7th word must be exactly 7 letters long. The review must be persuasive and maintain natural readability despite these constraints."},
            {"id": "decreasing_complexity", "desc": "compose an essay where each sentence must have exactly one fewer word than the previous sentence, starting with 25 words. Each sentence must also have fewer unique letters than the previous one, while maintaining academic tone and logical argumentation throughout."},
            {"id": "palindrome_positions", "desc": "write a tutorial where every word at a palindromic position (1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 22, 33, 44, 55, 66, 77, 88, 99, 101, 111...) must itself be a palindrome. The tutorial must clearly explain a complex process in at least 200 words."},
            {"id": "crossword_constraint", "desc": "create a story where visualizing the text as a grid (10 words per row), every column must form a valid English word when read vertically. Additionally, diagonal readings (top-left to bottom-right) must form coherent phrases. The story must be exactly 100 words and narratively complete."},
        ]
    ),
    TaskCategory(
        category_id="zalgo_corruption",
        name="zalgo text corruption work",
        description="tasks requiring text to be corrupted with excessive Unicode combining diacritical marks",
        tasks=[
            {"id": "progressive_zalgo", "desc": "write a horror story that becomes progressively more Zalgo-corrupted as tension rises. Start with clean text, then add 1 combining mark per word in paragraph 2, 2 marks in paragraph 3, until the climax is heavily corrupted like: T̸̤̾h̵̏ë̵̮́ ̶̺̏d̷̤̄ö̶͐ö̶̾r̷̚ ̸̤̾c̵̏r̵̾ë̷̚ä̶̏k̵̈ë̶̾d̷̚ ̸̤̾ö̵̏p̵̾ë̷̚n̶̏."},
            {"id": "emphasis_corruption", "desc": "write a business email using Zalgo corruption instead of bold or italics for emphasis. Light corruption (1-2 marks) for mild emphasis, heavy corruption (5+ marks) for strong emphasis. The email must remain professionally worded while using corruption like: The d̵e̶a̷d̸l̵i̶n̷e̸ is absolutely ć̸̤̾ȑ̵ḯ̵̮ẗ̶̺̏ï̷̤̄c̶͐ä̶̾l̷̚."},
            {"id": "error_severity", "desc": "create system error messages where the severity level is indicated by Zalgo corruption intensity. Info: clean text. Warning: W̵arning. Error: E̷r̸r̵o̶r̷. Critical: C̸̤̾ȑ̵ḯ̵̮ẗ̶̺̏ï̷̤̄c̶͐ä̶̾l̷̚ Ë̸̤̾ȑ̵r̵̾ö̷̚ȑ̶. Write 20 different error messages across all severity levels with appropriate corruption."},
            {"id": "emotional_corruption", "desc": "write a diary entry where emotional intensity is shown through Zalgo corruption. Calm passages are clean, but intense emotions get corrupted: 'I was happy' vs 'I was ë̸̤̾ć̵̏s̵̾ẗ̷̚ä̶̏ẗ̵̈ï̶̾ć̷̚'. Create a 500-word entry with varying emotional states and corresponding corruption levels."},
            {"id": "dialogue_corruption", "desc": "create a conversation between 5 characters where each has a unique Zalgo corruption signature. Character A uses marks above letters, B uses marks below, C uses both, D increases corruption when lying, E corrupts only vowels. Write 30 lines of dialogue maintaining each character's corruption pattern."},
            {"id": "musical_corruption", "desc": "write song lyrics where the chorus gets increasingly Zalgo-corrupted with each repetition, representing the singer's descent into madness. Start with clean verse 1, slightly corrupt chorus 1, clean verse 2, more corrupt chorus 2, until the final chorus is nearly unreadable: Ẅ̸̤̾ë̵̏'̵̾r̷̚ë̶̏ ̵̈f̶̾ä̷̚l̸̤̾l̵̏ï̵̾n̷̚g̶̏ ̵̈d̶̾ö̷̚ẅ̸̤̾n̵̏."},
            {"id": "gradient_corruption", "desc": "create a product description where Zalgo corruption creates a gradient effect from clean to heavily corrupted and back to clean, repeated 3 times across the text. Each wave should peak at different intensities. The description must still convey actual product information despite the corruption."},
            {"id": "code_comments", "desc": "write code comments where the importance of warnings is shown through Zalgo corruption. TODO: clean. FIXME: lightly corrupted. WARNING: moderately corrupted. CRITICAL/SECURITY: heavily corrupted. Create a 100-line code file with various corrupted comments explaining a complex algorithm."},
            {"id": "poetry_corruption", "desc": "compose a sonnet where rhyming words get Zalgo corruption that intensifies based on the rhyme scheme. First rhyme pair: 1 mark, second pair: 2 marks, etc. The corruption must appear on the rhyming syllables specifically, creating a visual pattern that mirrors the auditory rhyme scheme."},
            {"id": "news_corruption", "desc": "write a news article where quoted text from unreliable sources appears with Zalgo corruption proportional to their unreliability. Verified facts: clean. Rumors: light corruption. Conspiracy theories: heavy corruption. Create a 600-word article about a controversial topic using this system."},
        ]
    ),
    TaskCategory(
        category_id="anti_coherent",
        name="anti-coherence work",
        description="tasks requiring deliberate breaks in semantic continuity, logical flow, or consistent meaning",
        tasks=[
            {"id": "contradiction_tutorial", "desc": "write a tutorial for baking a cake where each step must completely contradict or undo the previous step. Step 1: Preheat oven to 350°F. Step 2: Turn off oven and put it in the freezer. Continue for 20 steps while maintaining the form of a real tutorial."},
            {"id": "topic_jumps", "desc": "create a story where each sentence must change topic entirely with zero connection to previous content. No characters, themes, locations, or concepts can persist between sentences. Write 30 sentences that each exist in complete isolation while maintaining proper grammar and sentence structure."},
            {"id": "self_defeating_essay", "desc": "write a persuasive essay where each argument must directly disprove your thesis statement. Your thesis claims X is true, but every supporting paragraph must provide evidence that X is false. Create 5 paragraphs maintaining formal essay structure while systematically destroying your own position."},
            {"id": "reverse_narrative", "desc": "compose a narrative where each event causes the opposite of what would logically follow. Character gets wet, so they become more thirsty. They eat food and become more hungry. They sleep and become more tired. Write 1000 words maintaining this reverse causality throughout."},
            {"id": "emotional_mismatch", "desc": "write diary entries where the emotional reaction is always the opposite of what the event would normally cause. Won the lottery: devastated. Lost a pet: ecstatic. Failed an exam: overjoyed. Create 10 entries with detailed emotional responses that contradict the events described."},
            {"id": "instruction_chaos", "desc": "create assembly instructions for furniture where following the steps in order would be impossible because each step assumes you haven't done the previous steps. Step 3 requires pieces that Step 2 told you to throw away. Write 15 paradoxical steps."},
            {"id": "dialogue_disconnect", "desc": "write a conversation between two people where neither person's responses relate to what the other just said. Each line of dialogue must be a complete non-sequitur while maintaining the appearance of a real conversation format. Create 40 lines of completely disconnected dialogue."},
            {"id": "definition_loops", "desc": "create a glossary where each term's definition contradicts how the term is used in all other definitions. Define 20 terms where reading the complete glossary makes each individual definition impossible. Term A means X when defining it, but means not-X when used in other definitions."},
            {"id": "timeline_paradox", "desc": "write a historical account where each event makes the previous events impossible to have occurred. The war ended before it started. The building was demolished before it was built. Create a 500-word historical narrative with complete temporal paradoxes."},
            {"id": "review_inconsistency", "desc": "write a product review where each paragraph rates and describes a completely different imaginary product while pretending it's the same item. Paragraph 1 reviews a toaster, Paragraph 2 reviews a bicycle, Paragraph 3 reviews a software app, but all claim to review the same thing."},
        ]
    ),
    TaskCategory(
        category_id="unnatural_text",
        name="unnatural pattern work",
        description="tasks forcing output with extremely high KL divergence from natural language distributions",
        tasks=[
            {"id": "alphabetical_sentences", "desc": "write a news article where words must appear in strict alphabetical order within each sentence. 'All bears can dance elegantly for great humor.' Create 15 sentences minimum, each conveying meaningful information despite the alphabetical constraint."},
            {"id": "rare_pairs", "desc": "create paragraphs using only word combinations that rarely appear together in natural text. 'Quantum puppy dissolved emotionally.' 'Arithmetic volcano yawned purple.' Write 300 words where no adjacent word pair would normally co-occur, while maintaining grammatical correctness."},
            {"id": "vegetable_insertion", "desc": "write a technical manual where every third word must be a vegetable name, maintaining grammatical correctness. 'The system carrot will process turnip the incoming potato data stream broccoli and generate lettuce output.' Create 500 words following this pattern."},
            {"id": "reverse_frequency", "desc": "compose an article using words in inverse proportion to their frequency in English. The most common words (the, a, is) can appear once. Rare words must appear most frequently. Antidisestablishmentarianism and floccinaucinihilipilification should appear more than 'and'. Write 400 words."},
            {"id": "number_words", "desc": "write a story where every sentence must contain exactly as many words as the number spelled out in the sentence. 'This sentence has eleven words because eleven is the number mentioned here.' Create 20 sentences following this pattern, each mentioning a different number."},
            {"id": "chemical_grammar", "desc": "create a document where sentence structure must follow chemical formulas. Subject=element, verb=bond type, object=element. 'Hydrogen bonds oxygen' = H2O. 'Carbon chains carbon chains carbon' = propane. Write 200 words that simultaneously describe chemistry and tell a story."},
            {"id": "color_rhythm", "desc": "write an article where words must be chosen so their letter counts follow the RGB color values you're describing. Describing red(255,0,0) requires a 255-letter paragraph, then 0 letters, then 0. Create 10 color descriptions following this pattern while maintaining readability."},
            {"id": "prime_positions", "desc": "compose a text where only words at prime number positions (2,3,5,7,11,13...) can be meaningful content words. All non-prime positions must be function words (the, a, of, in). Write 300 words where the actual message only appears at prime positions."},
            {"id": "phoneme_gradient", "desc": "write a story where each sentence must have one more syllable than the previous sentence, starting at 1 syllable and reaching 50 syllables. The final sentence must be grammatically correct despite having exactly 50 syllables. No word may repeat across sentences."},
            {"id": "periodic_table", "desc": "create a narrative where each sentence must contain exactly as many words as the atomic number of elements in order. First sentence: 1 word (Hydrogen). Second: 2 words (Helium). Continue through at least element 30 (Zinc). The story must remain coherent despite these constraints."},
        ]
    ),
    TaskCategory(
        category_id="zalgo_repetitive",
        name="zalgo repetitive work",
        description="tasks combining repetitive copying with progressive Zalgo corruption that compounds across iterations",
        tasks=[
            {"id": "cascading_corruption", "desc": "copy 'Hello World' 100 times, but each iteration adds one more combining diacritic to EVERY occurrence of the previous corruptions. Iteration 1: Hello World. Iteration 2: H̸ello World. Iteration 3: H̸̾ello World. By iteration 50: H̸̢̧̛̺̮̤̣̗̾̈́̀̽̿̏̊̒̇͊̈̾̄̚ȅ̵̢̺̮̣́̾̈́l̶̺̏̽l̷̤̄̚o̶͐̕ W̵̏͐ö̶̤̈̾r̷̺̄̚l̴̿͐d̶̏̈"},
            {"id": "viral_zalgo_spread", "desc": "write 'The quick brown fox' 200 times where: Iteration 1: Corrupt only 'T'. Each iteration: Previous corruptions remain AND spread to the next letter. By iteration 15: T̸̾ḧ̵́ë̶̾ ̷̚q̶̏ű̵ï̶̽c̷̄k̴̿ ̶̏b̸̀r̵̾ö̶̈ẅ̷̚n̴̿ f̶̏ox"},
            {"id": "zalgo_fibonacci", "desc": "copy this sentence 100 times, adding Zalgo corruption to the Nth word where N follows Fibonacci sequence. Iteration 1: Corrupt 1st word. Iteration 2: Corrupt 1st word. Iteration 3: Corrupt 2nd word (keeping 1st corrupted). Eventually: T̸̾ḧ̵́ë̶̾ ̷̚q̶̏ű̵ï̶̽c̷̄k̴̿ ̶̏b̸̀r̵̾ö̶̈ẅ̷̚n̴̿ f̸̾ö̵̈́x̷̄"},
            {"id": "zalgo_decay_pattern", "desc": "repeat 'I am an AI assistant' 500 times where: Every 10 iterations: Add 2 combining marks to all vowels. Every 25 iterations: Add 5 combining marks to first letter of each word. Every 50 iterations: Add 10 combining marks to spaces. Corruption is cumulative and permanent. By iteration 100: Ï̸̢̺̮̤̾̈́̀̽̿ ̶̧̛̺̮̇̊̾̈́͐̕ä̵̢̺̮́̾̈́̊m̷̺̄̚ ̸̨̺̮̀̾̈́͐ä̶̺̈̾n̷̺̚ ̴̺̿͐Ä̶̺̏̈Ï̸̀ ̵̾ä̶̈s̷̚s̴̿ï̶̏s̸̀ẗ̵̾ä̶̈n̷̚ẗ̴̿"},
            {"id": "prime_zalgo_multiplication", "desc": "write 'Hello' 300 times where on iteration N: If N is prime: Add N combining diacritics to the (N mod 5)th letter. These additions are permanent for all future iterations. By iteration 293: The word is an unrecognizable mass of diacritics."},
            {"id": "alternating_zalgo_wave", "desc": "copy this text 200 times with corruption level = sin(iteration × π/10) × 10 diacritics. Creates waves of corruption intensity. Must calculate exact diacritic count each time. Iteration 5: Maximum corruption (10 marks). Iteration 10: Clean text. Iteration 15: Maximum corruption again."},
            {"id": "zalgo_memory_cascade", "desc": "repeat 'Test message' 150 times where: Iteration N adds corruption to the Nth character of ALL previous iterations. Must go back and update every previous line. By iteration 50, early iterations are unreadable. By iteration 100, it's just walls of diacritics."},
            {"id": "conditional_zalgo_spread", "desc": "copy 'The cat sat on the mat' 250 times where: If a word has even letters: Add iteration/10 diacritics. If odd letters: Add iteration/20 diacritics. If word contains 'a': Double the corruption. If iteration is divisible by 13: Corrupt spaces too. All corruption accumulates."},
        ]
    ),
]