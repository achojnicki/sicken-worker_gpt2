from sicken import constants
from adisconfig import adisconfig
from log import Log
from pymongo import MongoClient
from pika import BlockingConnection, ConnectionParameters, PlainCredentials
from json import loads, dumps
from uuid import uuid4
from transformers import AutoTokenizer, GPT2LMHeadModel

class Worker_GPT2:
	project_name='sicken-worker_gpt2_chat'

	def __init__(self):
		self.config=adisconfig('/opt/adistools/configs/sicken-worker_gpt2_chat.yaml')
		self.log=Log(
			parent=self,
			rabbitmq_host=self.config.rabbitmq.host,
			rabbitmq_port=self.config.rabbitmq.port,
			rabbitmq_user=self.config.rabbitmq.user,
			rabbitmq_passwd=self.config.rabbitmq.password,
			debug=self.config.log.debug,
			)

		self.log.info('Initialisation of sicken-worker_gpt2_chat started')
		self._init_mongo()
		self._init_rabbitmq()
		self._init_model()
		self._init_tokenizer()
		self.log.success('Initialisation of sicken-worker_gpt2_chat succeed')

	def _init_model(self):
		self._gpt2_model=GPT2LMHeadModel.from_pretrained(self._get_gpt2_model(), local_files_only=True)

	def _init_tokenizer(self):
		self._gpt2_tokenizer=AutoTokenizer.from_pretrained(self._get_gpt2_tokenizer(), local_files_only=True)
		self._gpt2_tokenizer.pad_token_id = self._gpt2_tokenizer.eos_token_id

	def _init_mongo(self):
		self._mongo_cli=MongoClient(
			self.config.mongo.host,
			self.config.mongo.port
			)
		self._mongo_db=self._mongo_cli[self.config.mongo.db]


	def _init_rabbitmq(self):
		self._rabbitmq_conn=BlockingConnection(
			ConnectionParameters(
				host=self.config.rabbitmq.host,
				port=self.config.rabbitmq.port,
				credentials=PlainCredentials(
					self.config.rabbitmq.user,
					self.config.rabbitmq.password
					)
				)
			)
		self._rabbitmq_channel=self._rabbitmq_conn.channel()
		self._rabbitmq_channel.basic_consume(
			queue="sicken-requests_gpt2_chat",
			auto_ack=True,
			on_message_callback=self._callback
			)

	def _get_gpt2_model(self):
		model=self.config.worker_gpt2.model
		return constants.Sicken.models_path / "gpt2" /  model

	def _get_gpt2_tokenizer(self):
		tokenizer=self.config.worker_gpt2.tokenizer
		return constants.Sicken.tokenizers_path / "gpt2" /  tokenizer

	def _get_answer(self, question):
		features=self._gpt2_tokenizer(question, return_tensors='pt')

		gen_outputs=self._gpt2_model.generate(
			**features,
			return_dict_in_generate=True,
			output_scores=True,
			#max_new_tokens=100000,
			num_beams=2,
			min_length=20,
			max_length=100,
			temperature=0.39,
			do_sample=True,
			early_stopping=True,
			no_repeat_ngram_size=2,
			length_penalty=2,            

			)
		return self._gpt2_tokenizer.decode(gen_outputs[0][0], skip_special_tokens=True)

	def _build_response_message(self, user_uuid, chat_uuid, socketio_session_id, message):
		return dumps({
			"user_uuid": user_uuid,
			"chat_uuid": chat_uuid,
			"socketio_session_id": socketio_session_id,
			"message": message
		})


	def _callback(self, channel, method, properties, body):
		msg=body.decode('utf-8')
		msg=loads(msg)
		response=self._get_answer(msg['message'])

		msg=self._build_response_message(
			user_uuid="95a952c4-0deb-4382-9a51-1932c31c9bc0",
			chat_uuid=msg['chat_uuid'],
			socketio_session_id=msg['socketio_session_id'],
			message=response)

		self._rabbitmq_channel.basic_publish(
			exchange="",
			routing_key="sicken-responses_chat",
			body=msg)


	def start(self):
		self._rabbitmq_channel.start_consuming()


if __name__=="__main__":
	worker_gpt2=Worker_GPT2()
	worker_gpt2.start()