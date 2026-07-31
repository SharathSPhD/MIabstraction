{\rtf1\ansi\ansicpg1252\cocoartf2870
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\froman\fcharset0 Times-Roman;\f1\froman\fcharset0 Times-Bold;\f2\fmodern\fcharset0 Courier;
\f3\fnil\fcharset0 Menlo-Regular;\f4\froman\fcharset0 Times-Italic;}
{\colortbl;\red255\green255\blue255;\red0\green0\blue0;\red109\green109\blue109;\red0\green0\blue233;
}
{\*\expandedcolortbl;;\cssrgb\c0\c0\c0;\cssrgb\c50196\c50196\c50196;\cssrgb\c0\c0\c93333;
}
{\*\listtable{\list\listtemplateid1\listhybrid{\listlevel\levelnfc23\levelnfcn23\leveljc0\leveljcn0\levelfollow0\levelstartat0\levelspace360\levelindent0{\*\levelmarker \{disc\}}{\leveltext\leveltemplateid1\'01\uc0\u8226 ;}{\levelnumbers;}\fi-360\li720\lin720 }{\listname ;}\listid1}
{\list\listtemplateid2\listhybrid{\listlevel\levelnfc23\levelnfcn23\leveljc0\leveljcn0\levelfollow0\levelstartat0\levelspace360\levelindent0{\*\levelmarker \{disc\}}{\leveltext\leveltemplateid101\'01\uc0\u8226 ;}{\levelnumbers;}\fi-360\li720\lin720 }{\listname ;}\listid2}
{\list\listtemplateid3\listhybrid{\listlevel\levelnfc23\levelnfcn23\leveljc0\leveljcn0\levelfollow0\levelstartat0\levelspace360\levelindent0{\*\levelmarker \{disc\}}{\leveltext\leveltemplateid201\'01\uc0\u8226 ;}{\levelnumbers;}\fi-360\li720\lin720 }{\listname ;}\listid3}}
{\*\listoverridetable{\listoverride\listid1\listoverridecount0\ls1}{\listoverride\listid2\listoverridecount0\ls2}{\listoverride\listid3\listoverridecount0\ls3}}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\deftab720
\pard\pardeftab720\sa240\partightenfactor0

\f0\fs24 \cf0 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 I think this is one of the most important research questions in AI over the next decade.\
The short answer is:\
\pard\pardeftab720\sa240\partightenfactor0

\f1\b \cf0 Yes, but probably not in the way assembly programs a CPU.
\f0\b0 \
\pard\pardeftab720\sa240\partightenfactor0
\cf0 A more accurate analogy is that mechanistic interpretability could evolve into an 
\f1\b intermediate representation (IR) plus debugger plus compiler infrastructure
\f0\b0  for neural computation, rather than a fixed instruction set.\
That distinction is subtle but profound.\
\pard\pardeftab720\partightenfactor0
\cf3 \strokec3 \
\pard\pardeftab720\sa321\partightenfactor0

\f1\b\fs48 \cf0 \strokec2 Today's AI stack is "inside out"\
\pard\pardeftab720\sa240\partightenfactor0

\f0\b0\fs24 \cf0 Compare classical computing.\
\pard\pardeftab720\partightenfactor0

\f2\fs26 \cf0 Application\
\

\f3 \uc0\u8595 
\f2 \
\
Python\
\

\f3 \uc0\u8595 
\f2 \
\
C\
\

\f3 \uc0\u8595 
\f2 \
\
Compiler\
\

\f3 \uc0\u8595 
\f2 \
\
Assembly\
\

\f3 \uc0\u8595 
\f2 \
\
Instruction Set\
\

\f3 \uc0\u8595 
\f2 \
\
CPU\
\pard\pardeftab720\sa240\partightenfactor0

\f0\fs24 \cf0 Every layer is intentionally designed.\
Now compare LLMs.\
\pard\pardeftab720\partightenfactor0

\f2\fs26 \cf0 Application\
\

\f3 \uc0\u8595 
\f2 \
\
Prompt\
\

\f3 \uc0\u8595 
\f2 \
\
Tokenizer\
\

\f3 \uc0\u8595 
\f2 \
\
Transformer\
\

\f3 \uc0\u8595 
\f2 \
\
Linear Algebra\
\

\f3 \uc0\u8595 
\f2 \
\
GPU\
\pard\pardeftab720\sa240\partightenfactor0

\f0\fs24 \cf0 Notice what's missing.\
There is no\
\pard\tx220\tx720\pardeftab720\li720\fi-720\partightenfactor0
\ls1\ilvl0\cf0 \kerning1\expnd0\expndtw0 \outl0\strokewidth0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 variables\
\ls1\ilvl0\kerning1\expnd0\expndtw0 \outl0\strokewidth0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 functions\
\ls1\ilvl0\kerning1\expnd0\expndtw0 \outl0\strokewidth0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 objects\
\ls1\ilvl0\kerning1\expnd0\expndtw0 \outl0\strokewidth0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 memory model\
\ls1\ilvl0\kerning1\expnd0\expndtw0 \outl0\strokewidth0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 instruction set\
\ls1\ilvl0\kerning1\expnd0\expndtw0 \outl0\strokewidth0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 compiler\
\ls1\ilvl0\kerning1\expnd0\expndtw0 \outl0\strokewidth0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 debugger\
\pard\pardeftab720\sa240\partightenfactor0
\cf0 The transformer jumps directly from mathematics to application.\
That is incredibly unusual.\
\pard\pardeftab720\partightenfactor0
\cf3 \strokec3 \
\pard\pardeftab720\sa321\partightenfactor0

\f1\b\fs48 \cf0 \strokec2 Mechanistic interpretability is trying to create the missing middle\
\pard\pardeftab720\sa240\partightenfactor0

\f0\b0\fs24 \cf0 Suppose we eventually discover\
\pard\tx220\tx720\pardeftab720\li720\fi-720\partightenfactor0
\ls2\ilvl0\cf0 \kerning1\expnd0\expndtw0 \outl0\strokewidth0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 reusable features\
\ls2\ilvl0\kerning1\expnd0\expndtw0 \outl0\strokewidth0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 reusable circuits\
\ls2\ilvl0\kerning1\expnd0\expndtw0 \outl0\strokewidth0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 reusable memories\
\ls2\ilvl0\kerning1\expnd0\expndtw0 \outl0\strokewidth0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 reusable planning modules\
\pard\pardeftab720\sa240\partightenfactor0
\cf0 Then suddenly the stack becomes\
\pard\pardeftab720\partightenfactor0

\f2\fs26 \cf0 Application\
\

\f3 \uc0\u8595 
\f2 \
\
Prompt\
\

\f3 \uc0\u8595 
\f2 \
\
Representation Program\
\

\f3 \uc0\u8595 
\f2 \
\
Circuits\
\

\f3 \uc0\u8595 
\f2 \
\
Features\
\

\f3 \uc0\u8595 
\f2 \
\
Transformer\
\

\f3 \uc0\u8595 
\f2 \
\
Linear Algebra\
\

\f3 \uc0\u8595 
\f2 \
\
GPU\
\pard\pardeftab720\sa240\partightenfactor0

\f0\fs24 \cf0 Notice what happened.\
Mechanistic interpretability inserted an entire computational layer.\
\pard\pardeftab720\partightenfactor0
\cf3 \strokec3 \
\pard\pardeftab720\sa321\partightenfactor0

\f1\b\fs48 \cf0 \strokec2 We may stop "training models"\
\pard\pardeftab720\sa240\partightenfactor0

\f0\b0\fs24 \cf0 This is where your idea becomes exciting.\
Today we build LLMs like this.\
\pard\pardeftab720\partightenfactor0

\f2\fs26 \cf0 Data\
\

\f3 \uc0\u8595 
\f2 \
\
Gradient Descent\
\

\f3 \uc0\u8595 
\f2 \
\
Weights\
\

\f3 \uc0\u8595 
\f2 \
\
Model\
\pard\pardeftab720\sa240\partightenfactor0

\f0\fs24 \cf0 Almost no control.\
It's like breeding.\
\pard\pardeftab720\partightenfactor0
\cf3 \strokec3 \
\pard\pardeftab720\sa240\partightenfactor0
\cf0 \strokec2 Imagine instead\
\pard\pardeftab720\partightenfactor0

\f2\fs26 \cf0 Knowledge\
\

\f3 \uc0\u8595 
\f2 \
\
Feature Library\
\

\f3 \uc0\u8595 
\f2 \
\
Circuit Library\
\

\f3 \uc0\u8595 
\f2 \
\
Representation Compiler\
\

\f3 \uc0\u8595 
\f2 \
\
Weights\
\

\f3 \uc0\u8595 
\f2 \
\
Model\
\pard\pardeftab720\sa240\partightenfactor0

\f0\fs24 \cf0 Now we are engineering.\
Not breeding.\
\pard\pardeftab720\partightenfactor0
\cf3 \strokec3 \
\pard\pardeftab720\sa321\partightenfactor0

\f1\b\fs48 \cf0 \strokec2 Think about assembly language\
\pard\pardeftab720\sa240\partightenfactor0

\f0\b0\fs24 \cf0 Assembly says\
\pard\pardeftab720\partightenfactor0

\f2\fs26 \cf0 MOV AX,BX\
\
ADD AX,3\
\
JMP LOOP\
\pard\pardeftab720\sa240\partightenfactor0

\f0\fs24 \cf0 The CPU understands\
\pard\pardeftab720\partightenfactor0

\f2\fs26 \cf0 001011001011001\
\pard\pardeftab720\sa240\partightenfactor0

\f0\fs24 \cf0 The programmer never touches the binary.\
\pard\pardeftab720\partightenfactor0
\cf3 \strokec3 \
\pard\pardeftab720\sa240\partightenfactor0
\cf0 \strokec2 Now imagine future transformers.\
Instead of\
\pard\pardeftab720\partightenfactor0

\f2\fs26 \cf0 Fine tune on medical data.\
\pard\pardeftab720\sa240\partightenfactor0

\f0\fs24 \cf0 You write\
\pard\pardeftab720\partightenfactor0

\f2\fs26 \cf0 Install Medical Ontology\
\
Connect\
\
Anatomy Circuit\
\

\f3 \uc0\u8595 
\f2 \
\
Diagnostic Circuit\
\

\f3 \uc0\u8595 
\f2 \
\
Treatment Circuit\
\pard\pardeftab720\sa240\partightenfactor0

\f0\fs24 \cf0 The compiler transforms that into neural parameters.\
No one knows how to do this today.\
But conceptually it isn't crazy.\
\pard\pardeftab720\partightenfactor0
\cf3 \strokec3 \
\pard\pardeftab720\sa321\partightenfactor0

\f1\b\fs48 \cf0 \strokec2 Even more interesting...\
\pard\pardeftab720\sa240\partightenfactor0

\f0\b0\fs24 \cf0 Imagine programming the residual stream directly.\
Instead of\
\pard\pardeftab720\partightenfactor0

\f2\fs26 \cf0 Prompt\
\

\f3 \uc0\u8595 
\f2 \
\
Hope model reasons\
\pard\pardeftab720\sa240\partightenfactor0

\f0\fs24 \cf0 you write\
\pard\pardeftab720\partightenfactor0

\f2\fs26 \cf0 Working Memory\
\

\f3 \uc0\u8595 
\f2 \
\
Goal\
\

\f3 \uc0\u8595 
\f2 \
\
Planning\
\

\f3 \uc0\u8595 
\f2 \
\
Verification\
\

\f3 \uc0\u8595 
\f2 \
\
Output\
\pard\pardeftab720\sa240\partightenfactor0

\f0\fs24 \cf0 Almost like programming RAM.\
Representation engineering is already exploring direct interventions in activation space\'97adding, removing, or modifying directions in the residual stream to change behaviour without retraining. {\field{\*\fldinst{HYPERLINK "https://arxiv.org/html/2502.17601v1?utm_source=chatgpt.com"}}{\fldrslt \cf4 \ul \ulc4 \strokec4 arXiv}}\
\pard\pardeftab720\partightenfactor0
\cf3 \strokec3 \
\pard\pardeftab720\sa321\partightenfactor0

\f1\b\fs48 \cf0 \strokec2 Training itself changes\
\pard\pardeftab720\sa240\partightenfactor0

\f0\b0\fs24 \cf0 Today's optimisation\
\pard\pardeftab720\partightenfactor0

\f2\fs26 \cf0 Data\
\

\f3 \uc0\u8595 
\f2 \
\
Loss\
\

\f3 \uc0\u8595 
\f2 \
\
Gradient\
\

\f3 \uc0\u8595 
\f2 \
\
Weights\
\pard\pardeftab720\sa240\partightenfactor0

\f0\fs24 \cf0 Suppose mechanistic understanding matures.\
Training could become\
\pard\pardeftab720\partightenfactor0

\f2\fs26 \cf0 Desired capability\
\

\f3 \uc0\u8595 
\f2 \
\
Desired circuit\
\

\f3 \uc0\u8595 
\f2 \
\
Desired features\
\

\f3 \uc0\u8595 
\f2 \
\
Training objective\
\

\f3 \uc0\u8595 
\f2 \
\
Gradient\
\pard\pardeftab720\sa240\partightenfactor0

\f0\fs24 \cf0 Instead of hoping SGD discovers the right algorithm,\
you guide it.\
Almost like constraint-based synthesis.\
\pard\pardeftab720\partightenfactor0
\cf3 \strokec3 \
\pard\pardeftab720\sa321\partightenfactor0

\f1\b\fs48 \cf0 \strokec2 Imagine "imports"\
\pard\pardeftab720\sa240\partightenfactor0

\f0\b0\fs24 \cf0 Python\
\pard\pardeftab720\partightenfactor0

\f2\fs26 \cf0 import numpy\
\pard\pardeftab720\sa240\partightenfactor0

\f0\fs24 \cf0 Future LLM\
\pard\pardeftab720\partightenfactor0

\f2\fs26 \cf0 import arithmetic\
\
import chemistry\
\
import planning\
\
import legal_reasoning\
\pard\pardeftab720\sa240\partightenfactor0

\f0\fs24 \cf0 These aren't datasets.\
They're verified circuit libraries.\
\pard\pardeftab720\partightenfactor0
\cf3 \strokec3 \
\pard\pardeftab720\sa321\partightenfactor0

\f1\b\fs48 \cf0 \strokec2 Imagine linking\
\pard\pardeftab720\sa240\partightenfactor0

\f0\b0\fs24 \cf0 Today\
\pard\pardeftab720\partightenfactor0

\f2\fs26 \cf0 Everything\
\

\f3 \uc0\u8595 
\f2 \
\
One giant network\
\pard\pardeftab720\sa240\partightenfactor0

\f0\fs24 \cf0 Future\
\pard\pardeftab720\partightenfactor0

\f2\fs26 \cf0 Vision Module\
\

\f3 \uc0\u8595 
\f2 \
\
Shared Representation Bus\
\

\f3 \uc0\u8595 
\f2 \
\
Planning Module\
\

\f3 \uc0\u8595 
\f2 \
\
Language Module\
\

\f3 \uc0\u8595 
\f2 \
\
Action Module\
\pard\pardeftab720\sa240\partightenfactor0

\f0\fs24 \cf0 Exactly how software libraries are linked.\
\pard\pardeftab720\partightenfactor0
\cf3 \strokec3 \
\pard\pardeftab720\sa321\partightenfactor0

\f1\b\fs48 \cf0 \strokec2 Imagine debugging\
\pard\pardeftab720\sa240\partightenfactor0

\f0\b0\fs24 \cf0 Today\
\pard\pardeftab720\partightenfactor0

\f2\fs26 \cf0 Model hallucinated.\
\
Why?\
\
Unknown.\
\pard\pardeftab720\sa240\partightenfactor0

\f0\fs24 \cf0 Future\
\pard\pardeftab720\partightenfactor0

\f2\fs26 \cf0 Hallucination\
\

\f3 \uc0\u8595 
\f2 \
\
Locate Truth Circuit\
\

\f3 \uc0\u8595 
\f2 \
\
Inspect\
\

\f3 \uc0\u8595 
\f2 \
\
Feature corrupted\
\

\f3 \uc0\u8595 
\f2 \
\
Repair\
\

\f3 \uc0\u8595 
\f2 \
\
Done\
\pard\pardeftab720\sa240\partightenfactor0

\f0\fs24 \cf0 That begins to resemble symbolic debugging.\
\pard\pardeftab720\partightenfactor0
\cf3 \strokec3 \
\pard\pardeftab720\sa321\partightenfactor0

\f1\b\fs48 \cf0 \strokec2 Steering becomes programming\
\pard\pardeftab720\sa240\partightenfactor0

\f0\b0\fs24 \cf0 Today\
Steering looks primitive.\
\pard\pardeftab720\partightenfactor0

\f2\fs26 \cf0 Residual\
\
+\
\
Truth vector\
\pard\pardeftab720\sa240\partightenfactor0

\f0\fs24 \cf0 Tomorrow\
Imagine\
\pard\pardeftab720\partightenfactor0

\f2\fs26 \cf0 enable(\
    truthfulness,\
    mathematical_reasoning,\
    uncertainty_estimation\
)\
\
disable(\
    sycophancy,\
    memorization\
)\
\pard\pardeftab720\sa240\partightenfactor0

\f0\fs24 \cf0 Those calls could compile into activation interventions.\
We're nowhere near that level yet, but current work on representation engineering, steering vectors, and causal interventions is already exploring the first steps in this direction. {\field{\*\fldinst{HYPERLINK "https://arxiv.org/html/2502.17601v1?utm_source=chatgpt.com"}}{\fldrslt \cf4 \ul \ulc4 \strokec4 arXiv}}\
\pard\pardeftab720\partightenfactor0
\cf3 \strokec3 \
\pard\pardeftab720\sa321\partightenfactor0

\f1\b\fs48 \cf0 \strokec2 The compiler analogy becomes stronger\
\pard\pardeftab720\sa240\partightenfactor0

\f0\b0\fs24 \cf0 Today's compiler\
\pard\pardeftab720\partightenfactor0

\f2\fs26 \cf0 Python\
\

\f3 \uc0\u8595 
\f2 \
\
LLVM IR\
\

\f3 \uc0\u8595 
\f2 \
\
Assembly\
\

\f3 \uc0\u8595 
\f2 \
\
Machine Code\
\pard\pardeftab720\sa240\partightenfactor0

\f0\fs24 \cf0 Future\
\pard\pardeftab720\partightenfactor0

\f2\fs26 \cf0 Representation Language\
\

\f3 \uc0\u8595 
\f2 \
\
Circuit IR\
\

\f3 \uc0\u8595 
\f2 \
\
Feature Graph\
\

\f3 \uc0\u8595 
\f2 \
\
Transformer Parameters\
\pard\pardeftab720\sa240\partightenfactor0

\f0\fs24 \cf0 Notice\
there is still no symbolic execution inside the transformer.\
Only matrices.\
Exactly like CPUs ultimately execute voltages.\
\pard\pardeftab720\partightenfactor0
\cf3 \strokec3 \
\pard\pardeftab720\sa321\partightenfactor0

\f1\b\fs48 \cf0 \strokec2 The biggest conceptual leap\
\pard\pardeftab720\sa240\partightenfactor0

\f0\b0\fs24 \cf0 I think the real future is not\
\pard\pardeftab720\partightenfactor0

\f2\fs26 \cf0 Mechanistic Interpretability\
\pard\pardeftab720\sa240\partightenfactor0

\f0\fs24 \cf0 but\
\pard\pardeftab720\partightenfactor0

\f2\fs26 \cf0 Mechanistic Engineering\
\pard\pardeftab720\sa240\partightenfactor0

\f0\fs24 \cf0 Today\
\pard\pardeftab720\partightenfactor0

\f2\fs26 \cf0 Observe\
\
Interpret\
\
Explain\
\pard\pardeftab720\sa240\partightenfactor0

\f0\fs24 \cf0 Tomorrow\
\pard\pardeftab720\partightenfactor0

\f2\fs26 \cf0 Design\
\
Compose\
\
Compile\
\
Verify\
\
Repair\
\pard\pardeftab720\sa240\partightenfactor0

\f0\fs24 \cf0 That would transform the field from a science into an engineering discipline.\
\pard\pardeftab720\partightenfactor0
\cf3 \strokec3 \
\pard\pardeftab720\sa321\partightenfactor0

\f1\b\fs48 \cf0 \strokec2 This suggests a possible evolution\

\itap1\trowd \taflags0 \trgaph108\trleft-108 \trbrdrt\brdrnil \trbrdrl\brdrnil \trbrdrr\brdrnil 
\clvertalc \clshdrawnil \clwWidth1915\clftsWidth3 \clmart10 \clmarl10 \clmarb10 \clmarr10 \clbrdrt\brdrnil \clbrdrl\brdrnil \clbrdrb\brdrnil \clbrdrr\brdrnil \clpadt20 \clpadl20 \clpadb20 \clpadr20 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth5216\clftsWidth3 \clmart10 \clmarl10 \clmarb10 \clmarr10 \clbrdrt\brdrnil \clbrdrl\brdrnil \clbrdrb\brdrnil \clbrdrr\brdrnil \clpadt20 \clpadl20 \clpadb20 \clpadr20 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\qc\partightenfactor0

\fs24 \cf0 Era\cell 
\pard\intbl\itap1\pardeftab720\qc\partightenfactor0
\cf0 Primary activity\cell \row

\itap1\trowd \taflags0 \trgaph108\trleft-108 \trbrdrl\brdrnil \trbrdrr\brdrnil 
\clvertalc \clshdrawnil \clwWidth1915\clftsWidth3 \clmart10 \clmarl10 \clmarb10 \clmarr10 \clbrdrt\brdrnil \clbrdrl\brdrnil \clbrdrb\brdrnil \clbrdrr\brdrnil \clpadt20 \clpadl20 \clpadb20 \clpadr20 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth5216\clftsWidth3 \clmart10 \clmarl10 \clmarb10 \clmarr10 \clbrdrt\brdrnil \clbrdrl\brdrnil \clbrdrb\brdrnil \clbrdrr\brdrnil \clpadt20 \clpadl20 \clpadb20 \clpadr20 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0

\f0\b0 \cf0 2020\'962025\cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 Observe neurons and heads\cell \row

\itap1\trowd \taflags0 \trgaph108\trleft-108 \trbrdrl\brdrnil \trbrdrr\brdrnil 
\clvertalc \clshdrawnil \clwWidth1915\clftsWidth3 \clmart10 \clmarl10 \clmarb10 \clmarr10 \clbrdrt\brdrnil \clbrdrl\brdrnil \clbrdrb\brdrnil \clbrdrr\brdrnil \clpadt20 \clpadl20 \clpadb20 \clpadr20 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth5216\clftsWidth3 \clmart10 \clmarl10 \clmarb10 \clmarr10 \clbrdrt\brdrnil \clbrdrl\brdrnil \clbrdrb\brdrnil \clbrdrr\brdrnil \clpadt20 \clpadl20 \clpadb20 \clpadr20 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 2025\'962030\cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 Discover features and circuits\cell \row

\itap1\trowd \taflags0 \trgaph108\trleft-108 \trbrdrl\brdrnil \trbrdrr\brdrnil 
\clvertalc \clshdrawnil \clwWidth1915\clftsWidth3 \clmart10 \clmarl10 \clmarb10 \clmarr10 \clbrdrt\brdrnil \clbrdrl\brdrnil \clbrdrb\brdrnil \clbrdrr\brdrnil \clpadt20 \clpadl20 \clpadb20 \clpadr20 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth5216\clftsWidth3 \clmart10 \clmarl10 \clmarb10 \clmarr10 \clbrdrt\brdrnil \clbrdrl\brdrnil \clbrdrb\brdrnil \clbrdrr\brdrnil \clpadt20 \clpadl20 \clpadb20 \clpadr20 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 2030\'962035\cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 Build representation engineering tools\cell \row

\itap1\trowd \taflags0 \trgaph108\trleft-108 \trbrdrl\brdrnil \trbrdrt\brdrnil \trbrdrr\brdrnil 
\clvertalc \clshdrawnil \clwWidth1915\clftsWidth3 \clmart10 \clmarl10 \clmarb10 \clmarr10 \clbrdrt\brdrnil \clbrdrl\brdrnil \clbrdrb\brdrnil \clbrdrr\brdrnil \clpadt20 \clpadl20 \clpadb20 \clpadr20 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth5216\clftsWidth3 \clmart10 \clmarl10 \clmarb10 \clmarr10 \clbrdrt\brdrnil \clbrdrl\brdrnil \clbrdrb\brdrnil \clbrdrr\brdrnil \clpadt20 \clpadl20 \clpadb20 \clpadr20 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 2035+ (speculative)\cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 Compose, compile, verify, and debug neural programs\cell \lastrow\row
\pard\pardeftab720\partightenfactor0
\cf3 \strokec3 \
\pard\pardeftab720\sa321\partightenfactor0

\f1\b\fs48 \cf0 \strokec2 One caveat\'97and it is a big one\
\pard\pardeftab720\sa240\partightenfactor0

\f0\b0\fs24 \cf0 The analogy with CPUs breaks at one crucial point.\
A CPU was 
\f1\b designed around a fixed ISA
\f0\b0 .\
A transformer is 
\f1\b not
\f0\b0 .\
Each trained model may discover different internal circuits for the same external capability. Features may be distributed, circuits may be redundant, and multiple solutions can coexist. Current mechanistic interpretability research repeatedly emphasizes superposition, model-specific representations, and the challenge of finding abstractions that generalize across architectures. {\field{\*\fldinst{HYPERLINK "https://transformer-circuits.pub/?utm_source=chatgpt.com"}}{\fldrslt \cf4 \ul \ulc4 \strokec4 Transformer Circuits}}\
So the future programming layer may not target "attention head 17" or "feature 8129." Instead, it may target a higher-level, model-independent representation that is then 
\f4\i compiled
\f0\i0  into whatever internal circuits a particular model uses.\
That leads to a particularly interesting reframing:\
\pard\tx220\tx720\pardeftab720\li720\fi-720\partightenfactor0
\ls3\ilvl0
\f1\b \cf0 \kerning1\expnd0\expndtw0 \outl0\strokewidth0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 Transformer architecture
\f0\b0  becomes analogous to the CPU microarchitecture.\
\ls3\ilvl0
\f1\b \kerning1\expnd0\expndtw0 \outl0\strokewidth0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 Mechanistic interpretability
\f0\b0  discovers the computational semantics.\
\ls3\ilvl0
\f1\b \kerning1\expnd0\expndtw0 \outl0\strokewidth0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 Representation engineering
\f0\b0  becomes the first generation of "programming" against those semantics.\
\pard\tx220\tx720\pardeftab720\li720\fi-720\partightenfactor0
\ls3\ilvl0\cf0 \kerning1\expnd0\expndtw0 \outl0\strokewidth0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 A future 
\f1\b mechanistic compiler
\f0\b0  could map human-specified computational goals into trainable objectives, circuits, or activation-space interventions.\
\pard\pardeftab720\sa240\partightenfactor0
\cf0 If that vision materializes, we may look back at today's workflow\'97"collect more data, run gradient descent, hope the right circuits emerge"\'97the way we now look at programming computers by manually toggling front-panel switches before assembly language existed. The important uncertainty is not whether such an abstraction layer is conceivable; it's whether neural networks admit a stable enough computational structure for that layer to be both expressive and portable across models.\
}