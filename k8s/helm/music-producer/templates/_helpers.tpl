{{/*
Expand the name of the chart.
*/}}
{{- define "music-producer.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "music-producer.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "music-producer.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "music-producer.labels" -}}
helm.sh/chart: {{ include "music-producer.chart" . }}
{{ include "music-producer.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "music-producer.selectorLabels" -}}
app.kubernetes.io/name: {{ include "music-producer.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "music-producer.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "music-producer.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
API component labels
*/}}
{{- define "music-producer.api.labels" -}}
{{ include "music-producer.labels" . }}
app.kubernetes.io/component: api
{{- end }}

{{/*
API selector labels
*/}}
{{- define "music-producer.api.selectorLabels" -}}
{{ include "music-producer.selectorLabels" . }}
app.kubernetes.io/component: api
{{- end }}

{{/*
Worker component labels
*/}}
{{- define "music-producer.worker.labels" -}}
{{ include "music-producer.labels" . }}
app.kubernetes.io/component: worker
{{- end }}

{{/*
Worker selector labels
*/}}
{{- define "music-producer.worker.selectorLabels" -}}
{{ include "music-producer.selectorLabels" . }}
app.kubernetes.io/component: worker
{{- end }}

{{/*
Create image pull secrets
*/}}
{{- define "music-producer.imagePullSecrets" -}}
{{- with .Values.global.imagePullSecrets }}
imagePullSecrets:
{{- toYaml . | nindent 2 }}
{{- end }}
{{- end }}
