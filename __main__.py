from adistools.adisconfig import adisconfig
from adistools.log import Log
from sicken.sicken.gpt2 import Sicken

from pymongo import MongoClient
from pika import BlockingConnection, ConnectionParameters, PlainCredentials
from json import loads, dumps
from uuid import uuid4
from transformers import AutoTokenizer, GPT2LMHeadModel

class Worker_GPT2:
	project_name='sicken-worker_gpt2_chat'

	def __init__(self):
		self._config=adisconfig('/opt/adistools/configs/sicken-worker_gpt2.yaml')
		self._log=Log(
			parent=self,
			rabbitmq_host=self._config.rabbitmq.host,
			rabbitmq_port=self._config.rabbitmq.port,
			rabbitmq_user=self._config.rabbitmq.user,
			rabbitmq_passwd=self._config.rabbitmq.password,
			debug=self._config.log.debug,
			)

		self._log.info(f'Initialisation of {self.project_name} started')
	
		self._sicken=Sicken(
			root=self,
			model=self._config.worker_gpt2.model,
			tokenizer=self._config.worker_gpt2.tokenizer)

		self._mongo_cli=MongoClient(
			self._config.mongo.host,
			self._config.mongo.port
			)

		self._mongo_db=self._mongo_cli[self._config.mongo.db]
		
		self._rabbitmq_conn=BlockingConnection(
			ConnectionParameters(
				host=self._config.rabbitmq.host,
				port=self._config.rabbitmq.port,
				credentials=PlainCredentials(
					self._config.rabbitmq.user,
					self._config.rabbitmq.password
					)
				)
			)
		self._rabbitmq_channel=self._rabbitmq_conn.channel()
		self._rabbitmq_channel.basic_consume(
			queue="sicken-requests_gpt2",
			auto_ack=True,
			on_message_callback=self._callback
			)

		self._log.success(f'Initialisation of {self.project_name} succeed')
	


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
		response=self._sicken.get_answer(msg)

		msg=self._build_response_message(
			user_uuid="95a952c4-0deb-4382-9a51-1932c31c9bc0",
			chat_uuid=msg['chat_uuid'],
			socketio_session_id=msg['socketio_session_id'],
			message=response)

		self._rabbitmq_channel.basic_publish(
			exchange="",
			routing_key="sicken-responses",
			body=msg)


	def start(self):
		self._rabbitmq_channel.start_consuming()


if __name__=="__main__":
	worker_gpt2=Worker_GPT2()
	worker_gpt2.start()