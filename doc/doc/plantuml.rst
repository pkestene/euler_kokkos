PlantUML
========

PlantUML is a tool to generate diagrams from textual source code. It can be used
to generate UML-compliant diagrams, hence the name. Many non-UML diagrams are
also supported making it a very versatile tool.

PlantUML is actually a "drawing tool", meaning it can do more than only UML
diagrams. It will allow you to "draw" diagrams that are not legal UML.

Two short examples of PlantUML diagrams are shown below. The specification of
the language can be found at http://plantuml.com/ with plenty of examples.

.. uml::
	:caption: The harsh reality of being a thread.

	you -> glibc: pthread_kill_other_threads_np()
	you <- glibc: **deleted**

.. uml::
	:caption: Grade schooler's math.

	:<math>x^2+y_1+z_12^34</math>;
	note right: AsciiMath
	:<latex>\sum_{i=0}^{n-1} (a_i + b_i^2)</latex>;
	note right: <latex>\LaTeX</latex>
