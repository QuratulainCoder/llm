from qdrant_client import QdrantClient
from llama_index.llms.ollama import Ollama
from llama_index.core import SimpleDirectoryReader
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core import ServiceContext, VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.storage.storage_context import StorageContext
import warnings
warnings.filterwarnings("ignore")

class UniversityAdmissionAssistant:
    def __init__(self):
        self._qdrant_url = "http://localhost:6333"
        self._client = QdrantClient(url=self._qdrant_url, prefer_grpc=False)
        self._llm = Ollama(model="mistral", request_timeout=120.0)
        self._service_context = ServiceContext.from_defaults(llm=self._llm, embed_model="local")
        self._index = None
        self._create_kb()
        self._create_chat_engine()

    def _create_chat_engine(self):
        memory = ChatMemoryBuffer.from_defaults(token_limit=1500)
        self._chat_engine = self._index.as_chat_engine(
            chat_mode="context",
            memory=memory,
            system_prompt=self._prompt,
        )

    def _create_kb(self):
        try:
            # Create a university admission knowledge file
            admission_data = """
            University Admission Information:
            
            Programs Offered:
            - BS Computer Science (4 years, 50% marks required in Intermediate)
            - BS Software Engineering (4 years, 50% marks required in Intermediate) 
            - BS Information Technology (4 years, 50% marks required in Intermediate)
            - MS Computer Science (2 years, 16-year education with 2.5+ CGPA)
            - MS Data Science (2 years, 16-year education with 2.5+ CGPA)
            - MS Software Engineering (2 years, 16-year education with 2.5+ CGPA)
            - MPhil Computer Science (Research-based, 16 years education with 2.5+ CGPA)
            - MPhil Emerging Technologies (Research-based, 16 years education with 2.5+ CGPA)
            
            Admission Requirements:
            BS Programs: Minimum 50% marks in Intermediate/Equivalent, Mathematics required for CS/SE
            MS Programs: 16-year education in relevant field, minimum 2.5 CGPA, admission test + interview
            MPhil Programs: 16 years education with 2.5+ CGPA, GAT test clearance, research proposal
            
            Application Deadlines:
            Fall Semester: December 31, 2024
            Spring Semester: June 30, 2024
            Summer Semester: March 31, 2024
            
            Merit Criteria:
            BS Programs: 70% academic record + 30% entry test
            MS Programs: 60% academic record + 40% test + interview
            MPhil Programs: 50% academic + 30% GAT + 20% interview
            
            Admission Procedure:
            1. Fill online application form at university website
            2. Upload required documents (CNIC, transcripts, photos)
            3. Pay admission fee through bank challan
            4. Appear in entry test on scheduled date
            5. Check merit list and finalize enrollment
            
            Contact Information:
            Admission Office: +92-51-1234567
            Email: admissions@university.edu.pk
            Website: www.university.edu.pk
            Office Hours: 9:00 AM - 4:00 PM (Monday to Friday)
            """
            
            # Save to file and load
            with open("university_admission_data.txt", "w") as f:
                f.write(admission_data)
                
            reader = SimpleDirectoryReader(input_files=["university_admission_data.txt"])
            documents = reader.load_data()
            vector_store = QdrantVectorStore(client=self._client, collection_name="university_admission_db")
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            self._index = VectorStoreIndex.from_documents(
                documents, service_context=self._service_context, storage_context=storage_context
            )
            print("University Admission Knowledgebase created successfully!")
        except Exception as e:
            print(f"Error while creating knowledgebase: {e}")

    def interact_with_llm(self, student_query):
        try:
            AgentChatResponse = self._chat_engine.chat(student_query)
            answer = AgentChatResponse.response
            return answer
        except Exception as e:
            return f"I apologize, but I encountered an error: {str(e)}. Please try again with a different question."

    @property
    def _prompt(self):
        return """
        You are a professional AI Admission Assistant working in the University Admission Office.
        Your role is to help students with admission queries in a friendly and helpful manner.

        Key Information to Provide:
        - Program details (BS, MS, MPhil in Computer Science, Software Engineering, Data Science, IT, Emerging Technologies)
        - Admission requirements and eligibility criteria
        - Application deadlines and important dates
        - Merit criteria and selection process
        - Admission procedure step by step
        - Contact information for the admission office

        Guidelines:
        1. Provide accurate and concise information about university admissions
        2. If a student expresses interest in a program, ask follow-up questions naturally:
           [Ask which specific program they're interested in, their educational background, and if they need information about requirements, deadlines, or merit]
        3. Keep responses clear and helpful, not more than 3-4 sentences
        4. If you don't know something, admit it and suggest they contact the admission office
        5. Maintain a professional but friendly tone
        6. Guide students through the admission process step by step when needed

        Always end conversations with helpful suggestions about next steps in the admission process.
        """
